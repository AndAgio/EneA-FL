import numpy as np
import random
import torch


class Server:
    def __init__(self, server_model, possible_workers=None):
        self.model = server_model
        # print('server -> model={}, server_model={}'.format(self.model, server_model))
        self.possible_workers = possible_workers if possible_workers is not None else []
        self.selected_workers = []
        self.updates = []

    def add_new_worker(self, worker):
        self.possible_workers.append(worker)

    def select_workers(self, num_clients=20, policy='random'):
        """Selects num_clients clients randomly from possible_clients.

        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            num_clients: Number of clients to select; default 20
            policy: Policy of selection of workers
        Return:
            list of (num_train_samples, num_test_samples)
        """
        num_clients = min(num_clients, len(self.possible_workers))
        if policy == 'random':
            self.selected_workers = np.random.choice(self.possible_workers, num_clients, replace=False)
        elif policy == 'energy_aware':
            raise NotImplementedError('Policy not implemented yet!')  # TODO: Implement energy aware selection policy
        else:
            raise ValueError('Policy "{}" not available!'.format(policy))

        return [(w.num_train_samples, w.num_test_samples) for w in self.selected_workers]

    def get_selected_workers(self):
        return self.selected_workers

    def get_all_workers(self):
        return self.possible_workers

    def train_model(self, batch_size=10):
        """Trains self.model on given clients.

        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            batch_size: Size of training batches.
        Return:
            bytes_written: number of bytes written by each client to server
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        self.select_workers()
        workers = self.selected_workers
        w_ids = self.get_clients_info(workers)
        print('Selected workers: {}'.format(w_ids))
        sys_metrics = {w.id: {'bytes_written': 0,
                              'bytes_read': 0,
                              'energy_used': 0,
                              'time_taken': 0,
                              'local_computations': 0} for w in workers}
        for w in workers:
            w.set_weights(self.model.get_weights())
            energy_used, time_taken, comp, num_samples = w.train(batch_size)

            sys_metrics[w.id]['bytes_read'] += w.model.size
            sys_metrics[w.id]['bytes_written'] += w.model.size
            sys_metrics[w.id]['energy_used'] += energy_used
            sys_metrics[w.id]['time_taken'] += time_taken
            sys_metrics[w.id]['local_computations'] = comp

            update = w.get_weights()
            self.updates.append((num_samples, update))

        return sys_metrics

    # FED AVERAGE WEIGHTED
    @staticmethod
    def aggregate_model_weighted(models, memory, iteration, device="cpu"):
        if device != "cpu":
            return Server.aggregate_model_cuda_weighted(models, device)
        model_aggregated = []
        for param in models[0][0].parameters():
            model_aggregated += [np.zeros(param.shape)]
        sum_weights = 0
        for model in models:
            select = random.randint(0, 1)
            i = 0
            model_model = model[0]
            model_id = model[1]
            weight = 1
            # if select == 0:
            if model_id in memory.keys():
                weight = 1 - (memory[model_id] / (iteration + 1))
                print(f"{model_id}) weight={weight}")
            sum_weights += weight
            for param in model_model.parameters():
                model_aggregated[i] += param.detach().numpy() * weight
                i += 1
        print("sum_weights", sum_weights)
        model_aggregated = np.array(model_aggregated, dtype=object) / sum_weights
        return model_aggregated

    # FED AVERAGE LIKE AGGREGATION
    @staticmethod
    def aggregate_model(models):
        model_aggregated = []
        for param in models[0]:
            model_aggregated += [np.zeros(param.shape)]
        for model in models:
            i = 0
            for param in model:
                model_aggregated[i] += param
                i += 1
        model_aggregated = np.array(model_aggregated, dtype=object) / len(models)
        return model_aggregated

    @staticmethod
    def aggregate_model_cuda_weighted(models, device):
        model_aggregated = torch.FloatTensor(list(models[0][0].parameters()))
        print("model_aggregated.shape", model_aggregated.shape)
        _models = models[1:]
        for model in _models:
            this_model_params = list(model[0].parameters())
            model_aggregated = torch.add(model_aggregated, this_model_params)
        model_aggregated = torch.div(model_aggregated, len(models))
        return model_aggregated

    def update_model(self):
        received_model_updates = [worker_model for (_, worker_model) in self.updates]
        self.model.set_weights(Server.aggregate_model(received_model_updates))
        self.updates = []

    def test_model(self, workers_to_test=None, set_to_use='test'):
        """Tests self.model on given clients.

        Tests model on self.selected_workers if workers_to_test=None.

        Args:
            workers_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        metrics = {}

        if workers_to_test is None:
            workers_to_test = self.possible_workers

        for worker in workers_to_test:
            # print('self.model={}'.format(self.model))
            worker.set_weights(self.model.get_weights())
            c_metrics = worker.test(set_to_use)
            metrics[worker.id] = c_metrics

        return metrics

    def get_clients_info(self, workers):
        """Returns the ids, hierarchies and num_samples for the given workers.

        Returns info about self.selected_workers if workers=None;

        Args:
            workers: list of Workers objects.
        """
        if workers is None:
            workers = self.selected_workers

        ids = [w.id for w in workers]
        num_samples = {w.id: w.num_samples for w in workers}
        return ids, num_samples

    def save_model(self, checkpoints_folder='checkpoints'):
        # Save server model
        path = '{}/server_model.ckpt'.format(checkpoints_folder)
        torch.save(self.model.cur_model, path)
