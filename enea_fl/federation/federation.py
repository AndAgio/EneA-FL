import os
import numpy as np
from .server import Server
from .worker import Worker
from .utils import get_stat_writer_function, get_sys_writer_function, print_stats
from enea_fl.models import ServerModel, WorkerModel, read_data


class Federation:
    def __init__(self, dataset, n_rounds=100, use_val_set=False):
        self.dataset = dataset
        self.n_rounds = n_rounds
        self.use_val_set = use_val_set
        print('Setting up federation for learning over {} in {} rounds'.format(dataset.upper(), n_rounds))
        self.workers = Federation.setup_workers(dataset, use_val_set)
        self.server = Federation.create_server(dataset, self.workers)
        self.worker_ids, self.worker_num_samples = self.server.get_clients_info(self.workers)
        print('Federation initialized with {} workers!'.format(len(self.workers)))

    def run(self, clients_per_round=10, batch_size=10, eval_every=1):
        # Initial status
        print('--- Random Initialization ---')
        stat_writer_fn = get_stat_writer_function(self.worker_ids, self.worker_num_samples,
                                                  metrics_dir='metrics', metrics_name='federation')
        sys_writer_fn = get_sys_writer_function(metrics_dir='metrics', metrics_name='federation')
        print_stats(0, self.server, stat_writer_fn, self.use_val_set)

        # Simulate training
        for i in range(self.n_rounds):
            print('--- Round {} of {}: Training {} workers ---'.format(i + 1, self.n_rounds, clients_per_round))

            # Simulate server model training on selected clients' data
            sys_metrics = self.server.train_model(batch_size=batch_size)
            worker_ids, worker_num_samples = self.server.get_clients_info(self.server.get_selected_workers())
            sys_writer_fn(i + 1, worker_ids, sys_metrics, worker_num_samples)

            # Update server model
            self.server.update_model()

            # Test model
            if (i + 1) % eval_every == 0 or (i + 1) == self.n_rounds:
                print_stats(i + 1, self.server, stat_writer_fn, self.use_val_set)

        # Save server model
        ckpt_path = os.path.join('checkpoints', self.dataset)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        save_path = self.server.save_model(checkpoints_folder=ckpt_path)
        print('Model saved in path: %s' % save_path)

    @staticmethod
    def create_workers(users, device_types, energy_policies, train_data, test_data, dataset):
        workers = [Worker(u, device_types[i], energy_policies[i],
                          train_data[u], test_data[u], WorkerModel(dataset)) for i, u in enumerate(users)]
        return workers

    @staticmethod
    def create_server(dataset, possible_workers):
        print('Setting up server...')
        return Server(ServerModel(dataset), possible_workers)

    @staticmethod
    def setup_workers(dataset, use_val_set=False):
        """Instantiates workers based on given train and test data directories.

        Return:
            all_workers: list of Client objects.
        """
        print('Setting up workers...')
        eval_set = 'test' if not use_val_set else 'val'
        train_data_dir = os.path.join('data', dataset, 'data', 'train')
        test_data_dir = os.path.join('data', dataset, 'data', eval_set)
        users, _, train_data, test_data = read_data(train_data_dir, test_data_dir)
        device_types = np.random.choice(['raspberry_0', 'raspberry_2', 'raspberry_3', 'nano', 'xavier'],
                                        size=len(users), replace=True)
        energy_policies = np.random.choice(['normal', 'conservative', 'extreme'],
                                           size=len(users), replace=True)
        workers = Federation.create_workers(users, device_types, energy_policies, train_data, test_data, dataset)
        return workers
