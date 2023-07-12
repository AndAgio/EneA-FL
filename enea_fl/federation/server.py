import numpy as np
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
            self.logger.print_it('Selecting workers based on random policy!')
            self.selected_workers = np.random.choice(self.possible_workers, num_workers, replace=False)
        elif policy == 'energy_aware':
            self.logger.print_it('Selecting workers based on energy policy!')
            self.select_workers_based_on_energy(num_workers=num_workers, alpha=alpha, beta=beta, k=k)
        else:
            raise ValueError('Policy "{}" not available!'.format(policy))

        return [(w.num_train_samples, w.num_test_samples) for w in self.selected_workers]

    def select_workers_based_on_energy(self, num_workers=20, alpha=0.5, beta=0.5, k=0.9):
        assert alpha + beta == 1 and alpha >= 0 and beta >= 0
        last_workers = [w_id for (w_id, _, _) in self.last_updates]
        if len(last_workers) == 1:
            energy_workers = last_workers
            n_random_workers = num_workers - len(energy_workers)
            possible_other_workers = [worker for worker in self.possible_workers if worker not in energy_workers]
            other_workers = np.random.choice(possible_other_workers, n_random_workers, replace=False).tolist()
            self.selected_workers = energy_workers + other_workers
        else:
            n_energy_workers = math.floor(len(last_workers) * k)
            n_random_workers = num_workers - n_energy_workers
            metrics = {w_id: None for w_id in last_workers}
            acc_with_all = self.compute_accuracy_with_all_updates()
            for w_id in last_workers:
                metrics[w_id] = self.compute_metric(identity=w_id,
                                                    alpha=alpha,
                                                    beta=beta,
                                                    accuracy_with_all=acc_with_all)
            metrics = dict(sorted(metrics.items(), key=lambda item: item[1]))
            energy_workers_ids = list(list(metrics.keys())[:n_energy_workers])
            energy_workers = [self.get_worker_by_id(w_id) for w_id in energy_workers_ids]
            possible_other_workers = [worker for worker in self.possible_workers if worker not in energy_workers]
            other_workers = np.random.choice(possible_other_workers, n_random_workers, replace=False).tolist()
            self.selected_workers = energy_workers + other_workers

    def compute_metric(self, identity, alpha=0.5, beta=0.5, accuracy_with_all=None):
        if accuracy_with_all is None:
            raise ValueError('Cannot compute metric without knowing model accuracy with all updates!')
        model_updates = [worker_model for (w_id, _, worker_model) in self.last_updates if w_id != identity]
        # self.logger.print_it('Computing accuracy with all updates except one. '
        #                      'Updates length: {}'.format(len(model_updates)))
        model_with_all_but_one = self.model.create_copy()
        model_with_all_but_one.set_weights(Server.aggregate_model(model_updates))
        accuracy_without_id = model_with_all_but_one.test(test_data=self.local_test_data)['accuracy']
        del model_with_all_but_one
        num = accuracy_with_all - accuracy_without_id
        # Denominator
        energies_used = [self.last_iteration_consumption[w_id]['energy_used']
                         for w_id in list(self.last_iteration_consumption.keys())]
        times_taken = [self.last_iteration_consumption[w_id]['time_taken']
                       for w_id in list(self.last_iteration_consumption.keys())]
        max_energy = max(energies_used)
        max_time = max(times_taken)
        energy_used = self.last_iteration_consumption[identity]['energy_used']
        time_taken = self.last_iteration_consumption[identity]['time_taken']
        den = alpha * energy_used / max_energy + beta * time_taken / max_time
        metric = num / den
        dev_type = self.get_worker_by_id(identity).device_type
        ene_pol = self.get_worker_by_id(identity).energy_policy
        self.logger.print_it('Worker with identity "{}" is a {} with {} local energy policy.\n'
                             'It used {:.3f} Joules and took {:.3f} seconds to train.\n'
                             'The accuracy without him is {:.3f} and with him is {:.3f}.\n'
                             'Therefore, its energy effectiveness score is {:.3f}'.format(identity,
                                                                                          dev_type.upper(),
                                                                                          ene_pol.upper(),
                                                                                          energy_used,
                                                                                          time_taken,
                                                                                          accuracy_without_id * 100,
                                                                                          accuracy_with_all * 100,
                                                                                          metric))
        return metric

    def compute_accuracy_with_all_updates(self):
        received_model_updates = [worker_model for (_, _, worker_model) in self.last_updates]
        # self.logger.print_it('Computing accuracy with all updates. '
        #                      'Updates length: {}'.format(len(received_model_updates)))
        model_with_all = self.model.create_copy()
        model_with_all.set_weights(Server.aggregate_model(received_model_updates))
        accuracy_with_all = model_with_all.test(test_data=self.local_test_data)['accuracy']
        del model_with_all
        return accuracy_with_all

    def get_selected_workers(self):
        return self.selected_workers

    def get_all_workers(self):
        return self.possible_workers

    def get_worker_by_id(self, identity):
        for worker in self.possible_workers:
            if worker.id == identity:
                return worker
        raise ValueError('Worker with ID: {} not found!'.format(identity))

    def train_model(self, num_workers=10, batch_size=10, lr=0.1, round_ind=-1,
                    policy='energy_aware', alpha=0.5, beta=0.5, k=0.9,
                    max_update_latency=None):
        self.logger.print_it(' Round {} '.format(round_ind).center(60, '-'))
        _ = self.select_workers(num_workers=num_workers,
                                policy=policy if round_ind > 1 else 'random',
                                alpha=alpha,
                                beta=beta,
                                k=k)
        workers = self.selected_workers
        w_ids = self.get_clients_info(workers)
        self.logger.print_it('Selected workers: {}'.format(w_ids))
        sys_metrics = {w.id: {'bytes_written': 0,
                              'bytes_read': 0,
                              'energy_used': 0,
                              'time_taken': 0,
                              'local_computations': 0} for w in workers}

        self.last_iteration_consumption = {w.id: {'energy_used': 0,
                                                  'time_taken': 0} for w in workers}

        self.last_updates = []

        def train_worker(worker):
            worker.set_weights(self.model.get_weights())
            executed, energy_used, time_taken, comp, num_samples = worker.train(batch_size=batch_size,
                                                                                lr=lr,
                                                                                round_ind=round_ind)
            sys_metrics[worker.id]['bytes_read'] += worker.model.size
            sys_metrics[worker.id]['bytes_written'] += worker.model.size
            sys_metrics[worker.id]['energy_used'] += energy_used
            self.last_iteration_consumption[worker.id]['energy_used'] = energy_used
            sys_metrics[worker.id]['time_taken'] += time_taken
            self.last_iteration_consumption[worker.id]['time_taken'] = time_taken \
                if (max_update_latency is not None and time_taken < max_update_latency) or max_update_latency is None \
                else max_update_latency
            sys_metrics[worker.id]['local_computations'] = comp
            if (executed and max_update_latency is None) or (executed and time_taken < max_update_latency):
                update = worker.get_weights()
                self.updates.append((worker.id, num_samples, update))
                self.last_updates.append((worker.id, num_samples, update))
            else:
                self.logger.print_it('Worker {} did not execute the training in time! '
                                     'Either dead or too slow!'.format(worker.id))

        Parallel(n_jobs=len(workers), prefer="threads")(delayed(train_worker)(worker) for worker in workers)

        self.logger.print_it(''.center(60, '-'))
        return sys_metrics

    # FED AVERAGE LIKE AGGREGATION
    @staticmethod
    def aggregate_model(models_state_dict):
        state_dict_keys = list(models_state_dict[0].keys())
        aggregated_dict = {}
        for key in state_dict_keys:
            aggregated_dict[key] = torch.stack([model_dict[key].float() for model_dict in models_state_dict], 0).mean(0)
        return aggregated_dict

    # WEIGHTED FED AVERAGE LIKE AGGREGATION
    @staticmethod
    def aggregate_model_with_weights(models_state_dict, models_weight):
        state_dict_keys = list(models_state_dict[0].keys())
        aggregated_dict = {}
        for key in state_dict_keys:
            aggregated_dict[key] = torch.stack([model_dict[key].float() * models_weight[i]
                                                for i, model_dict in enumerate(models_state_dict)], 0).mean(0)
        return aggregated_dict

    def update_model(self):
        received_model_updates = [worker_model for (_, _, worker_model) in self.updates]
        self.model.set_weights(Server.aggregate_model(received_model_updates))
        self.updates = []

    def test_model_on_workers(self, workers_to_test=None, set_to_use='test', round_ind=-1, batch_size=10):
        metrics = {}
        if workers_to_test is None:
            workers_to_test = self.possible_workers

        def test_worker(worker):
            worker.set_weights(self.model.get_weights())
            c_metrics = worker.test_global(self.get_model(), set_to_use, round_ind=round_ind, batch_size=batch_size)
            metrics[worker.id] = c_metrics

        Parallel(n_jobs=len(workers_to_test), prefer="threads")(delayed(test_worker)(worker)
                                                                for worker in workers_to_test)
        return metrics

    def test_model_on_server(self, batch_size=10):
        metrics = self.model.test(test_data=self.local_test_data, batch_size=batch_size)
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
        return path

    def get_model(self):
        return self.model.model

    def get_server_model(self):
        return self.model
