import numpy as np
import copy
import math
import torch
from joblib import Parallel, delayed
from enea_fl.utils import DumbLogger


class Server:
    def __init__(self, server_model, possible_workers=None, test_data=None, logger=None):
        self.model = server_model
        self.possible_workers = possible_workers if possible_workers is not None else []
        self.selected_workers = []
        self.last_updates = []
        self.updates = []
        self.last_iteration_consumption = {}
        self.local_test_data = test_data
        self.logger = logger if logger is not None else DumbLogger()

    def add_new_worker(self, worker):
        self.possible_workers.append(worker)

    def select_workers(self, num_workers=20, policy='random', alpha=0.5, beta=0.5, k=0.9):
        num_workers = min(num_workers, len(self.possible_workers))
        if policy == 'random':
            self.selected_workers = np.random.choice(self.possible_workers, num_workers, replace=False)
        elif policy == 'energy_aware':
            self.select_workers_based_on_energy(num_workers=num_workers, alpha=alpha, beta=beta, k=k)
        else:
            raise ValueError('Policy "{}" not available!'.format(policy))

        return [(w.num_train_samples, w.num_test_samples) for w in self.selected_workers]

    def select_workers_based_on_energy(self, num_workers=20, alpha=0.5, beta=0.5, k=0.9):
        assert alpha + beta == 1 and alpha >= 0 and beta >= 0
        n_energy_workers = math.floor(num_workers * k)
        n_random_workers = num_workers - n_energy_workers
        last_workers = list(self.last_iteration_consumption.keys())
        metrics = {w_id: None for w_id in last_workers}
        for w_id in last_workers:
            metrics[w_id] = self.compute_metric(identity=w_id, alpha=alpha, beta=beta)
        metrics = dict(sorted(metrics.items(), key=lambda item: item[1]))
        energy_workers_ids = list(metrics.keys()[:n_energy_workers])
        energy_workers = [self.get_worker_by_id(w_id) for w_id in energy_workers_ids]
        other_workers = np.random.choice(self.possible_workers, n_random_workers, replace=False).to_list()
        self.selected_workers = energy_workers + other_workers

    def compute_metric(self, identity, alpha=0.5, beta=0.5, accuracy_with_all=None):
        if accuracy_with_all is None:
            received_model_updates = [worker_model for (_, _, worker_model) in self.updates]
            model = copy.deepcopy(self.model).set_weights(Server.aggregate_model(received_model_updates))
            accuracy_with_all = model.test(test_data=self.local_test_data)['accuracy']
        model_updates = [worker_model for (w_id, _, worker_model) in self.updates if w_id != identity]
        model = copy.deepcopy(self.model).set_weights(Server.aggregate_model(model_updates))
        accuracy_without_id = model.test(test_data=self.local_test_data)['accuracy']
        num = accuracy_with_all - accuracy_without_id
        den = alpha * self.last_iteration_consumption[identity]['energy_used'] + \
              beta * self.last_iteration_consumption[identity]['time_taken']
        return num / den

    def get_selected_workers(self):
        return self.selected_workers

    def get_all_workers(self):
        return self.possible_workers

    def get_worker_by_id(self, identity):
        for worker in self.possible_workers:
            if worker.id == identity:
                return worker
        raise ValueError('Worker with ID: {} not found!'.format(identity))

    def train_model(self, num_workers=10, batch_size=10, lr=0.1, round_ind=-1, alpha=0.5, beta=0.5, k=0.9):
        _ = self.select_workers(num_workers=num_workers,
                                policy='energy',
                                alpha=alpha,
                                beta=beta,
                                k=k)
        workers = self.selected_workers
        w_ids = self.get_clients_info(workers)
        self.logger.print_it(' Round {} '.format(round_ind).center(60, '-'))
        self.logger.print_it('Selected workers: {}'.format(w_ids))
        sys_metrics = {w.id: {'bytes_written': 0,
                              'bytes_read': 0,
                              'energy_used': 0,
                              'time_taken': 0,
                              'local_computations': 0} for w in workers}

        self.last_iteration_consumption = {w.id: {'energy_used': 0,
                                                  'time_taken': 0} for w in workers}

        def train_worker(worker):
            worker.set_weights(self.model.get_weights())
            energy_used, time_taken, comp, num_samples = worker.train(batch_size=batch_size,
                                                                      lr=lr,
                                                                      round_ind=round_ind)
            sys_metrics[worker.id]['bytes_read'] += worker.model.size
            sys_metrics[worker.id]['bytes_written'] += worker.model.size
            sys_metrics[worker.id]['energy_used'] += energy_used
            self.last_iteration_consumption[worker.id]['energy_used'] = energy_used
            sys_metrics[worker.id]['time_taken'] += time_taken
            self.last_iteration_consumption[worker.id]['time_taken'] = time_taken
            sys_metrics[worker.id]['local_computations'] = comp
            update = worker.get_weights()
            self.updates.append((worker.id, num_samples, update))

        Parallel(n_jobs=len(workers), prefer="threads")(delayed(train_worker)(worker) for worker in workers)

        self.logger.print_it('Obtained metrics: {}'.format(sys_metrics))
        self.logger.print_it(''.center(60, '-'))
        return sys_metrics

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

    def update_model(self):
        received_model_updates = [worker_model for (_, worker_model) in self.updates]
        self.model.set_weights(Server.aggregate_model(received_model_updates))
        self.updates = []

    def test_model(self, workers_to_test=None, set_to_use='test', round_ind=-1):
        metrics = {}
        if workers_to_test is None:
            workers_to_test = self.possible_workers

        def test_worker(worker):
            worker.set_weights(self.model.get_weights())
            c_metrics = worker.test_global(self.get_model(), set_to_use, round_ind=round_ind)
            metrics[worker.id] = c_metrics

        Parallel(n_jobs=len(workers_to_test), prefer="threads")(delayed(test_worker)(worker)
                                                                for worker in workers_to_test)

        return metrics

    def get_clients_info(self, workers):
        if workers is None:
            workers = self.selected_workers
        ids = [w.id for w in workers]
        num_samples = {w.id: w.num_samples for w in workers}
        return ids, num_samples

    def save_model(self, checkpoints_folder='checkpoints'):
        # Save server model
        path = '{}/server_model.ckpt'.format(checkpoints_folder)
        torch.save(self.model.cur_model, path)

    def get_model(self):
        return self.model.model

    def get_server_model(self):
        return self.model
