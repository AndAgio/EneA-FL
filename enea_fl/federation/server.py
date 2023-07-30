import numpy as np
import math
from scipy.special import softmax
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

    def select_workers(self, num_workers=20, policy='random', alpha=0.5, beta=0.5, k=0.9, round_ind=-1):
        num_workers = min(num_workers, len(self.possible_workers))
        if policy == 'random':
            self.logger.print_it('Selecting workers based on random policy!')
            self.selected_workers = np.random.choice(self.possible_workers, num_workers, replace=False)
        elif policy == 'acc_aware':
            self.logger.print_it('Selecting workers based on accuracy policy!')
            self.select_workers_based_on_accuracy(num_workers=num_workers, k=k)
        elif policy == 'energy_aware':
            self.logger.print_it('Selecting workers based on energy policy!')
            # self.select_workers_based_on_energy(num_workers=num_workers, alpha=alpha, beta=beta, k=k)
            self.select_workers_based_on_energy_history(num_workers=num_workers, alpha=alpha, beta=beta, k=k,
                                                        round_ind=round_ind)
        else:
            raise ValueError('Policy "{}" not available!'.format(policy))

        return [(w.num_train_samples, w.num_test_samples) for w in self.selected_workers]

    # def select_workers_based_on_energy(self, num_workers=20, alpha=0.5, beta=0.5, k=0.9):
    #     assert alpha + beta == 1 and alpha >= 0 and beta >= 0
    #     last_workers = [w_id for (w_id, _, _) in self.last_updates]
    #     if len(last_workers) == 1:
    #         energy_workers = last_workers
    #         n_random_workers = num_workers - len(energy_workers)
    #         possible_other_workers = [worker for worker in self.possible_workers if worker not in energy_workers]
    #         other_workers = np.random.choice(possible_other_workers, n_random_workers, replace=False).tolist()
    #         self.selected_workers = energy_workers + other_workers
    #     else:
    #         n_energy_workers = math.floor(len(last_workers) * k)
    #         n_random_workers = num_workers - n_energy_workers
    #         metrics = {w_id: None for w_id in last_workers}
    #         acc_with_all = self.compute_accuracy_with_all_updates()
    #         for w_id in last_workers:
    #             metrics[w_id] = self.compute_energy_eff_metric(identity=w_id,
    #                                                            accuracy_with_all=acc_with_all,
    #                                                            alpha=alpha,
    #                                                            beta=beta)
    #         metrics = dict(sorted(metrics.items(), key=lambda item: item[1]))
    #         self.selected_workers = self.select_workers_from_metrics(metrics=metrics,
    #                                                                  n_best_workers=n_energy_workers,
    #                                                                  n_random_workers=n_random_workers)

    # def compute_energy_eff_metric(self, identity, accuracy_with_all, alpha=0.5, beta=0.5):
    #     num, accuracy_without_id = self.compute_acc_diff(identity=identity,
    #                                                      accuracy_with_all=accuracy_with_all)
    #     # Denominator
    #     energies_used = [self.last_iteration_consumption[w_id]['energy_used']
    #                      for w_id in list(self.last_iteration_consumption.keys())]
    #     times_taken = [self.last_iteration_consumption[w_id]['time_taken']
    #                    for w_id in list(self.last_iteration_consumption.keys())]
    #     max_energy = max(energies_used)
    #     max_time = max(times_taken)
    #     energy_used = self.last_iteration_consumption[identity]['energy_used']
    #     time_taken = self.last_iteration_consumption[identity]['time_taken']
    #     den = alpha * energy_used / max_energy + beta * time_taken / max_time
    #     metric = num / den
    #     dev_type = self.get_worker_by_id(identity).device_type
    #     ene_pol = self.get_worker_by_id(identity).energy_policy
    #     self.logger.print_it('Worker with identity "{}" is a {} with {} local energy policy.\n'
    #                          'It used {:.3f} KJ and took {:.3f} seconds to train.\n'
    #                          'The accuracy without him is {:.3f} and with him is {:.3f}.\n'
    #                          'Therefore, its energy effectiveness score is {:.3f}'.format(identity,
    #                                                                                       dev_type.upper(),
    #                                                                                       ene_pol.upper(),
    #                                                                                       energy_used/(1000*1000),
    #                                                                                       time_taken,
    #                                                                                       accuracy_without_id * 100,
    #                                                                                       accuracy_with_all * 100,
    #                                                                                       metric))
    #     return metric

    def select_workers_based_on_energy_history(self, num_workers=20, alpha=0.5, beta=0.5, k=0.9, round_ind=-1):
        assert alpha >= 0 and beta >= 0
        self.compute_and_update_acc_diff_for_all_workers(round_ind=round_ind)
        metrics = {}
        for worker in self.possible_workers:
            metrics[worker.id] = self.compute_energy_time_acc_metric(identity=worker.id,
                                                                     alpha=alpha,
                                                                     beta=beta)
        metrics_avg = np.mean([val for val in list(metrics.values()) if val != 0 and val != np.inf])
        for worker in self.possible_workers:
            if metrics[worker.id] == 0:
                metrics[worker.id] = metrics_avg
        metrics = dict(sorted(metrics.items(), key=lambda item: item[1]))
        for w_id, metric in metrics.items():
            self.logger.print_it('Worker with identity "{}" has effectiveness score of {:.3f}'.format(w_id,
                                                                                                      metric))

        metrics = dict(sorted(metrics.items(), key=lambda item: item[1], reverse=False))
        n_energy_workers = math.floor(num_workers * k)
        n_random_workers = num_workers - n_energy_workers
        self.selected_workers = self.select_workers_from_metrics(metrics=metrics,
                                                                 n_best_workers=n_energy_workers,
                                                                 n_random_workers=n_random_workers,
                                                                 compute_p=True)

    def compute_energy_time_acc_metric(self, identity, alpha=0.5, beta=0.5):
        if self.get_worker_by_id(identity).get_tot_rounds_enrolled() == 0:
            dev_type = self.get_worker_by_id(identity).device_type
            ene_pol = self.get_worker_by_id(identity).energy_policy
            self.logger.print_it('Worker with identity "{}" is a {} with {} local energy policy.\n'
                                 'It has never been selected for a federation round!'.format(identity,
                                                                                             dev_type.upper(),
                                                                                             ene_pol.upper()))
            return 0
        else:
            energies_used = {w.id: w.get_energies_consumed() for w in self.possible_workers}
            times_taken = {w.id: w.get_times_taken() for w in self.possible_workers}
            rounds = [list(energy_consumed.keys()) for _, energy_consumed in energies_used.items()]
            rounds = sorted(list(set([r for device in rounds for r in device])))
            tot_metric = 0.
            for round_ind in rounds:
                in_round_energies_used = {w.id: energies_used[w.id].get(round_ind, 0.) for w in self.possible_workers}
                in_round_time_taken = {w.id: times_taken[w.id].get(round_ind, 0.) for w in self.possible_workers}
                max_energy = max(list(in_round_energies_used.values()))
                max_time = max(list(in_round_time_taken.values()))
                try:
                    energy_used = energies_used[identity][round_ind]
                    time_taken = times_taken[identity][round_ind]
                    if energy_used != 0 and time_taken != 0:
                        acc_diff = self.get_worker_by_id(identity).get_acc_diff_history()[round_ind]
                        tot_metric += alpha * energy_used / max_energy + \
                                      (1 - alpha) * time_taken / max_time - \
                                      beta * acc_diff
                    else:
                        tot_metric += 5
                except KeyError:
                    pass
            metric = tot_metric / self.get_worker_by_id(identity).get_tot_rounds_enrolled()
            dev_type = self.get_worker_by_id(identity).device_type
            ene_pol = self.get_worker_by_id(identity).energy_policy
            energy_history = self.get_worker_by_id(identity).get_energies_consumed()
            time_history = self.get_worker_by_id(identity).get_times_taken()
            acc_diff_history = self.get_worker_by_id(identity).get_acc_diff_history()
            self.logger.print_it('Worker with identity "{}" is a {} with {} local energy policy.\n'
                                 'Its energy history is {}.\n'
                                 'Its time history is {}.\n'
                                 'Its accuracy differential history is {}.\n'
                                 'Therefore, its energy effectiveness score is {:.3f}'.format(identity,
                                                                                              dev_type.upper(),
                                                                                              ene_pol.upper(),
                                                                                              energy_history,
                                                                                              time_history,
                                                                                              acc_diff_history,
                                                                                              metric))
            return metric

    def select_workers_based_on_accuracy(self, num_workers=20, k=0.9):
        last_workers = [w_id for (w_id, _, _) in self.last_updates]
        if len(last_workers) == 1:
            acc_workers = last_workers
            n_random_workers = num_workers - len(acc_workers)
            possible_other_workers = [worker for worker in self.possible_workers if worker not in acc_workers]
            other_workers = np.random.choice(possible_other_workers, n_random_workers, replace=False).tolist()
            self.selected_workers = acc_workers + other_workers
        else:
            n_acc_workers = math.floor(len(last_workers) * k)
            n_random_workers = num_workers - n_acc_workers
            metrics = {w_id: None for w_id in last_workers}
            acc_with_all = self.compute_accuracy_with_all_updates()
            for w_id in last_workers:
                metrics[w_id] = self.compute_accuracy_eff_metric(identity=w_id,
                                                                 accuracy_with_all=acc_with_all)
            metrics = dict(sorted(metrics.items(), key=lambda item: item[1], reverse=True))
            self.selected_workers = self.select_workers_from_metrics(metrics=metrics,
                                                                     n_best_workers=n_acc_workers,
                                                                     n_random_workers=n_random_workers)

    def compute_accuracy_eff_metric(self, identity, accuracy_with_all):
        metric, accuracy_without_id = self.compute_acc_diff(identity=identity,
                                                            accuracy_with_all=accuracy_with_all)
        dev_type = self.get_worker_by_id(identity).device_type
        ene_pol = self.get_worker_by_id(identity).energy_policy
        self.logger.print_it('Worker with identity "{}" is a {} with {} local energy policy.\n'
                             'The accuracy without him is {:.3f} and with him is {:.3f}.\n'
                             'Therefore, its accuracy effectiveness score is {:.3f}'.format(identity,
                                                                                            dev_type.upper(),
                                                                                            ene_pol.upper(),
                                                                                            accuracy_without_id * 100,
                                                                                            accuracy_with_all * 100,
                                                                                            metric))
        return metric

    def select_workers_from_metrics(self, metrics, n_best_workers, n_random_workers, compute_p=False):
        best_workers_ids = list(list(metrics.keys())[:n_best_workers])
        best_workers = [self.get_worker_by_id(w_id) for w_id in best_workers_ids]
        possible_other_workers = [worker for worker in self.possible_workers if worker not in best_workers]
        if compute_p:
            rounds_used = {worker.id: worker.get_tot_rounds_enrolled() + 1 for worker in possible_other_workers}
            tot_rounds = sum(list(rounds_used.values()))
            p = softmax([tot_rounds / round_wid for wid, round_wid in rounds_used.items()])
        else:
            p = None
        other_workers = np.random.choice(possible_other_workers, n_random_workers, p=p, replace=False).tolist()
        return best_workers + other_workers

    def compute_acc_diff(self, identity, accuracy_with_all):
        model_updates = [worker_model for (w_id, _, worker_model) in self.last_updates if w_id != identity]
        model_with_all_but_one = self.model.create_copy()
        model_with_all_but_one.set_weights(Server.aggregate_model(model_updates))
        accuracy_without_id = model_with_all_but_one.test(test_data=self.local_test_data)['accuracy']
        del model_with_all_but_one
        metric = accuracy_with_all - accuracy_without_id
        return metric, accuracy_without_id

    def compute_accuracy_with_all_updates(self):
        received_model_updates = [worker_model for (_, _, worker_model) in self.last_updates]
        # self.logger.print_it('Computing accuracy with all updates. '
        #                      'Updates length: {}'.format(len(received_model_updates)))
        model_with_all = self.model.create_copy()
        model_with_all.set_weights(Server.aggregate_model(received_model_updates))
        accuracy_with_all = model_with_all.test(test_data=self.local_test_data)['accuracy']
        del model_with_all
        return accuracy_with_all

    def compute_and_update_acc_diff_for_all_workers(self, round_ind):
        acc_with_all = self.compute_accuracy_with_all_updates()
        last_workers = [w_id for (w_id, _, _) in self.last_updates]
        for w_id in last_workers:
            acc_diff, _ = self.compute_acc_diff(identity=w_id, accuracy_with_all=acc_with_all)
            self.get_worker_by_id(w_id).update_acc_diff_history(round_ind=round_ind-1,
                                                                acc_diff=acc_diff)

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
                                k=k,
                                round_ind=round_ind)
        workers = self.selected_workers
        # w_ids = self.get_clients_info(workers)
        # self.logger.print_it('Selected workers: {}'.format(w_ids))
        self.logger.print_it('Selected workers: {}'.format(self.get_devices_and_samples_info(workers)))
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

    def send_model_to_all_workers(self):
        for worker in self.possible_workers:
            worker.set_weights(self.model.get_weights())

    def test_model_on_workers(self, workers_to_test=None, set_to_use='test', round_ind=-1, batch_size=10):
        metrics = {}
        if workers_to_test is None:
            workers_to_test = self.possible_workers

        def test_worker(worker):
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

    def get_devices_and_samples_info(self, workers):
        if workers is None:
            workers = self.selected_workers
        ids = [w.id for w in workers]
        infos = {w.id: (w.device_type, w.num_samples) for w in workers}
        return infos

    def save_model(self, checkpoints_folder='checkpoints'):
        # Save server model
        path = '{}/server_model.ckpt'.format(checkpoints_folder)
        torch.save(self.get_model(), path)
        return path

    def get_model(self):
        return self.model.model

    def get_server_model(self):
        return self.model
