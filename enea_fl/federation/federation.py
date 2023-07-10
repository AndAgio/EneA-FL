import os
import subprocess
import shutil
import numpy as np
import math
from .server import Server
from .worker import Worker
from .utils import print_workers_metrics, print_server_metrics, write_metrics_to_csv
from enea_fl.models import ServerModel, WorkerModel, read_data
from enea_fl.models.utils import get_word_emb_arr
from enea_fl.utils import get_logger


class Federation:
    def __init__(self, dataset, n_workers, max_spw=math.inf, sampling_mode='iid+sim', n_rounds=100, use_val_set=False):
        self.dataset = dataset
        self.n_workers = n_workers
        self.max_spw = max_spw
        self.sampling_mode = sampling_mode
        self.n_rounds = n_rounds
        self.use_val_set = use_val_set
        self.clean_previous_loggers()
        self.federation_logger = get_logger(node_type='federation',
                                            node_id='federation',
                                            log_folder=os.path.join('logs',
                                                                    dataset,
                                                                    '{}_workers'.format(n_workers),
                                                                    'spw={}'.format(max_spw),
                                                                    'mode={}'.format(sampling_mode)))
        self.federation_logger.print_it('Setting up federation for learning '
                                        'over {} in {} rounds'.format(dataset.upper(), n_rounds))
        self.workers = self.setup_workers()
        self.server = self.create_server()
        self.worker_ids, self.worker_num_samples = self.server.get_clients_info(self.workers)
        self.federation_logger.print_it('Federation initialized with {} workers!'.format(len(self.workers)))

    def run(self, clients_per_round=10, batch_size=10, lr=0.1, eval_every=1,
            policy='energy_aware', alpha=0.5, beta=0.5, k=0.9):
        # Initial status
        self.federation_logger.print_it(' Random Initialization '.center(60, '-'))
        self.test_workers_and_server(round_ind=0)

        # Simulate training
        for round_ind in range(self.n_rounds):
            self.federation_logger.print_it(' Round {} of {}: Training {} workers '.format(round_ind + 1,
                                                                                           self.n_rounds,
                                                                                           clients_per_round)
                                            .center(60, '-'))

            # Simulate server model training on selected clients' data
            sys_metrics = self.server.train_model(num_workers=clients_per_round,
                                                  batch_size=batch_size,
                                                  lr=lr,
                                                  round_ind=round_ind + 1,
                                                  policy=policy,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  k=k)
            worker_ids, worker_num_samples = self.server.get_clients_info(self.server.get_selected_workers())
            write_metrics_to_csv(num_round=round_ind + 1,
                                 ids=worker_ids,
                                 metrics=sys_metrics,
                                 partition='train',
                                 metrics_dir='metrics',
                                 metrics_name='{}_{}'.format('federation', 'energy'))

            # Update server model
            self.server.update_model()

            # Test model
            if (round_ind + 1) % eval_every == 0 or (round_ind + 1) == self.n_rounds:
                # print_stats(self.federation_logger, i + 1, self.server, stat_writer_fn, self.use_val_set)
                self.test_workers_and_server(round_ind=round_ind + 1)

        self.federation_logger.print_it(' Federation rounds finished! '.center(60, '-'))
        # Save server model
        ckpt_path = os.path.join('checkpoints', self.dataset)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        save_path = self.server.save_model(checkpoints_folder=ckpt_path)
        self.federation_logger.print_it('Model saved in path: {}'.format(save_path))

    def test_workers_and_server(self, round_ind):
        test_metrics = self.server.test_model_on_workers(set_to_use='test' if not self.use_val_set else 'val',
                                                         round_ind=round_ind)
        print_workers_metrics(logger=self.federation_logger,
                              metrics=test_metrics,
                              weights=self.server.get_clients_info(self.server.get_all_workers())[1],
                              prefix='{}_'.format('test' if not self.use_val_set else 'val'))
        server_test_metrics = self.server.test_model_on_server()
        print_server_metrics(logger=self.federation_logger,
                             metrics={'server': server_test_metrics})
        test_metrics['server'] = server_test_metrics
        write_metrics_to_csv(num_round=round_ind,
                             ids=[w.id for w in self.server.get_all_workers()] + ['server'],
                             metrics=test_metrics,
                             partition='test' if not self.use_val_set else 'val',
                             metrics_dir='metrics',
                             metrics_name='{}_{}'.format('federation', 'performance'))

    @staticmethod
    def create_workers(workers, device_types, energy_policies, train_data, test_data, dataset,
                       loggers=None, indexization=None):
        workers = [Worker(worker_id=u,
                          device_type=device_types[i],
                          energy_policy=energy_policies[i],
                          train_data=train_data[u],
                          eval_data=test_data[u],
                          model=WorkerModel(dataset=dataset, indexization=indexization),
                          logger=loggers[i]) for i, u in enumerate(workers)]
        return workers

    def create_server(self):
        self.federation_logger.print_it('Setting up server...')
        logger = get_logger(node_type='server',
                            node_id='server',
                            log_folder=os.path.join('logs',
                                                    self.dataset,
                                                    '{}_workers'.format(self.n_workers),
                                                    'spw={}'.format(self.max_spw),
                                                    'mode={}'.format(self.sampling_mode)))
        if self.dataset == 'sent140':
            indexization = self.gather_indexization()
        else:
            indexization = None
        # Read data for server node as if it was a single worker to get server data
        eval_set = 'test' if not self.use_val_set else 'val'
        try:
            self.federation_logger.print_it('Trying to read data from files...')
            _, _, test_data = Federation.read_data_from_dir(dataset=self.dataset,
                                                            n_workers=1,
                                                            sampling_mode='iid+sim',
                                                            max_spw=10000,
                                                            eval_set=eval_set)
        except (FileNotFoundError, AssertionError) as error:
            self.federation_logger.print_it('Files not found, processing data...')
            sf = 0.1 if self.dataset == 'femnist' else 1
            parent_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
            dataset_dir = os.path.join(parent_path, 'data', self.dataset)
            if isinstance(error, AssertionError):
                shutil.rmtree(os.path.join(dataset_dir, 'data', '1_workers',
                                           'spw=10000',
                                           'mode=iid+sim', 'sampled_data'))
                shutil.rmtree(os.path.join(dataset_dir, 'data', '1_workers',
                                           'spw=10000',
                                           'mode=iid+sim', 'train'))
                shutil.rmtree(os.path.join(dataset_dir, 'data', '1_workers',
                                           'spw=10000',
                                           'mode=iid+sim', 'test'))
            _ = subprocess.call("{}/preprocess.sh -s {} "
                                "--iu {} --spw {} --sf {}".format(dataset_dir,
                                                                  'iid+sim',
                                                                  1,
                                                                  10000,
                                                                  sf),
                                cwd=dataset_dir,
                                shell=True)
            _, _, test_data = Federation.read_data_from_dir(dataset=self.dataset,
                                                            n_workers=1,
                                                            sampling_mode='iid+sim',
                                                            max_spw=10000,
                                                            eval_set=eval_set)
        return Server(server_model=ServerModel(self.dataset, indexization=indexization),
                      possible_workers=self.workers,
                      test_data=test_data['0'],
                      logger=logger)

    def setup_workers(self):
        self.federation_logger.print_it('Setting up workers...')
        eval_set = 'test' if not self.use_val_set else 'val'
        try:
            self.federation_logger.print_it('Trying to read data from files...')
            workers_ids, train_data, test_data = Federation.read_data_from_dir(dataset=self.dataset,
                                                                               n_workers=self.n_workers,
                                                                               sampling_mode=self.sampling_mode,
                                                                               max_spw=self.max_spw,
                                                                               eval_set=eval_set)
        except (FileNotFoundError, AssertionError) as error:
            self.federation_logger.print_it('Files not found, processing data...')
            sf = 0.1 if self.dataset == 'femnist' else 1
            parent_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
            dataset_dir = os.path.join(parent_path, 'data', self.dataset)
            if isinstance(error, AssertionError):
                shutil.rmtree(os.path.join(dataset_dir, 'data', '{}_workers'.format(self.n_workers),
                                           'spw={}'.format(self.max_spw),
                                           'mode={}'.format(self.sampling_mode), 'sampled_data'))
                shutil.rmtree(os.path.join(dataset_dir, 'data', '{}_workers'.format(self.n_workers),
                                           'spw={}'.format(self.max_spw),
                                           'mode={}'.format(self.sampling_mode), 'train'))
                shutil.rmtree(os.path.join(dataset_dir, 'data', '{}_workers'.format(self.n_workers),
                                           'spw={}'.format(self.max_spw),
                                           'mode={}'.format(self.sampling_mode), 'test'))
            _ = subprocess.call("{}/preprocess.sh -s {} "
                                "--iu {} --spw {} --sf {}".format(dataset_dir,
                                                                  self.sampling_mode,
                                                                  self.n_workers,
                                                                  self.max_spw,
                                                                  sf),
                                cwd=dataset_dir,
                                shell=True)
            workers_ids, train_data, test_data = Federation.read_data_from_dir(dataset=self.dataset,
                                                                               n_workers=self.n_workers,
                                                                               sampling_mode=self.sampling_mode,
                                                                               max_spw=self.max_spw,
                                                                               eval_set=eval_set)
        device_types = np.random.choice(['raspberrypi', 'nano_cpu', 'nano_gpu', 'orin_cpu', 'orin_gpu'],
                                        size=len(workers_ids), replace=True)
        # energy_policies = np.random.choice(['normal', 'conservative', 'extreme'],
        #                                    size=len(workers), replace=True)
        energy_policies = np.random.choice(['normal'],
                                           size=len(workers_ids), replace=True)
        loggers = [get_logger(node_type='worker',
                              node_id=w,
                              log_folder=os.path.join('logs',
                                                      self.dataset,
                                                      '{}_workers'.format(self.n_workers),
                                                      'spw={}'.format(self.max_spw),
                                                      'mode={}'.format(self.sampling_mode)))
                   for w in workers_ids]
        if self.dataset == 'sent140':
            indexization = self.gather_indexization()
        else:
            indexization = None
        workers = Federation.create_workers(workers_ids, device_types, energy_policies,
                                            train_data, test_data, self.dataset, loggers, indexization)

        for i, u in enumerate(workers_ids):
            self.federation_logger.print_it('Device {} is a {} '
                                            'with {} local energy policy'.format(u,
                                                                                 device_types[i].upper(),
                                                                                 energy_policies[i].upper()))
        return workers

    @staticmethod
    def read_data_from_dir(dataset, n_workers=100, max_spw=math.inf, sampling_mode='iid+sim', eval_set='test'):
        train_data_dir = os.path.join('data', dataset, 'data', '{}_workers'.format(n_workers),
                                      'spw={}'.format(max_spw),
                                      'mode={}'.format(sampling_mode), 'train')
        test_data_dir = os.path.join('data', dataset, 'data', '{}_workers'.format(n_workers),
                                     'spw={}'.format(max_spw),
                                     'mode={}'.format(sampling_mode), eval_set)
        workers, _, train_data, test_data = read_data(train_data_dir, test_data_dir)
        assert len(workers) == n_workers
        return workers, train_data, test_data

    def clean_previous_loggers(self):
        log_folder = os.path.join('logs',
                                  self.dataset,
                                  '{}_workers'.format(self.n_workers),
                                  'spw={}'.format(self.max_spw),
                                  'mode={}'.format(self.sampling_mode))
        if os.path.exists(log_folder):
            shutil.rmtree(log_folder)

    def gather_indexization(self):
        self.federation_logger.print_it('Reading word indexization from GloVe\'s json...')
        try:
            _, indd, _ = get_word_emb_arr('enea_fl/models/embs.json')
        except FileNotFoundError:
            _ = subprocess.call("./enea_fl/models/get_embs.sh", shell=True)
            _, indd, _ = get_word_emb_arr('enea_fl/models/embs.json')
        return indd
