import math
import random

import torch
import numpy as np
from scipy.stats import expon

from enea_fl.utils import DumbLogger, compute_total_number_of_flops, \
    read_device_behaviours, get_average_energy, compute_avg_std_time_per_sample
from enea_fl.models import FemnistModel, MnistModel


class Worker:
    def __init__(self,
                 worker_id,
                 device_type='raspberrypi',
                 energy_policy='normal',
                 train_data={'x': [], 'y': []},
                 eval_data={'x': [], 'y': []},
                 model=None,
                 random_death=True,
                 cuda_device='cpu',
                 logger=None):
        self.logger = logger if logger is not None else DumbLogger()
        self.model = model
        self.model.set_logger(logger)
        self.id = worker_id
        self.device_type = device_type
        # gpu = True if device_type in ['nano', 'jetson'] else False
        self.processing_device = cuda_device
        self.logger.print_it('Worker {} is running on a {} and using '
                             'a {} for training and inference!'.format(self.id,
                                                                       self.device_type,
                                                                       self.processing_device))
        self.move_model_to_device()
        self.energy_policy = energy_policy
        self.train_data = train_data
        self.eval_data = eval_data

        self.tot_used_energy = 0.
        self.tot_rounds_enrolled = 0
        if random_death:
            mean_available_rounds = random.randint(0, 150)
            self.available_rounds = math.ceil(expon.rvs(scale=mean_available_rounds, size=1).item())
            self.logger.print_it('Worker {} will switch off after {} rounds!'.format(self.id,
                                                                                     self.available_rounds))
        else:
            self.available_rounds = np.inf

    def train(self, batch_size=10, lr=0.1, round_ind=-1):
        if self.is_dead():
            self.logger.print_it(' DEVICE IS DEAD! IT WILL NOT TRAIN THE MODEL! '.center(60, '-'))
            return False, 0, 0, 0, 0
        else:
            self.logger.print_it(' Training model at round {} '.format(round_ind).center(60, '-'))
            train_steps = self.compute_local_energy_policy(batch_size=batch_size)
            metrics = self.model.train(train_data=self.train_data,
                                       train_steps=train_steps,
                                       batch_size=batch_size,
                                       lr=lr)
            num_train_samples = train_steps * batch_size
            energy_used, time_taken = self.compute_consumed_energy_and_time(n_samples=num_train_samples)
            comp = compute_total_number_of_flops(model=self.model.model,
                                                 batch_size=batch_size)

            self.tot_used_energy += energy_used
            self.tot_rounds_enrolled += 1
            self.check_death()

            return True, energy_used, time_taken, comp, num_train_samples

    def compute_consumed_energy_and_time(self, n_samples):
        if isinstance(self.model.model, FemnistModel):
            dataset = 'femnist'
        elif isinstance(self.model.model, MnistModel):
            dataset = 'mnist'
        else:
            dataset = 'sent140'
        device_behaviour_files = read_device_behaviours(device_type=self.device_type,
                                                        dataset=dataset)
        dataset_size = len(self.train_data['y'])
        avg_energy, std_energy = get_average_energy(device_behaviour_files,
                                                    dataset_size=dataset_size,
                                                    dataset=dataset)
        avg_time_per_sample, std_time_per_sample = compute_avg_std_time_per_sample(device_behaviour_files,
                                                                                   dataset_size=dataset_size,
                                                                                   dataset=dataset)
        tot_energy = 0.
        tot_time = 0.
        for i in range(n_samples):
            sample_energy = np.random.normal(avg_energy, std_energy)
            sample_time = np.random.normal(avg_time_per_sample, std_time_per_sample)
            tot_energy += sample_energy * sample_time
            tot_time += sample_time
        return tot_energy, tot_time

    def test_local(self, set_to_use='test', round_ind=-1, batch_size=10):
        self.logger.print_it(' Testing local model at round {} '.format(round_ind).center(60, '-'))
        data = self.select_data_for_testing(set_to_use)
        return self.model.test_my_model(test_data=data, batch_size=batch_size)

    def test_global(self, model_to_test, set_to_use='test', round_ind=-1, batch_size=10):
        self.logger.print_it(' Testing global model at round {} '.format(round_ind).center(60, '-'))
        data = self.select_data_for_testing(set_to_use)
        return self.model.test_final_model(final_model=model_to_test,
                                           test_data=data,
                                           batch_size=batch_size)

    def select_data_for_testing(self, set_to_use='test'):
        assert set_to_use in ['train', 'test', 'val']
        if set_to_use == 'train':
            data = self.train_data
        elif set_to_use == 'test' or set_to_use == 'val':
            data = self.eval_data
        else:
            raise ValueError('Something wrong with data in testing!')
        return data

    def check_death(self):
        if self.tot_rounds_enrolled >= self.available_rounds:
            self.kill()
            return True
        else:
            return False

    def kill(self):
        self.energy_policy = 'extreme'

    def is_dead(self):
        if self.energy_policy == 'extreme':
            return True
        else:
            return False

    def compute_local_energy_policy(self, batch_size=10):
        epochs = 5
        if self.energy_policy == 'normal':
            return math.ceil((epochs * self.num_train_samples) / batch_size)
        elif self.energy_policy == 'conservative':
            return math.floor(((epochs * self.num_train_samples) / batch_size) / 10)
        elif self.energy_policy == 'extreme':
            return 0
        else:
            raise ValueError('Energy policy "{}" is not available!'.format(self.energy_policy))

    def set_weights(self, aggregated_numpy):
        self.model.set_weights(aggregated_numpy)

    def get_weights(self):
        return self.model.get_weights()

    @property
    def num_test_samples(self):
        if self.eval_data is None:
            return 0
        return len(self.eval_data['y'])

    @property
    def num_train_samples(self):
        if self.train_data is None:
            return 0
        return len(self.train_data['y'])

    @property
    def num_samples(self):
        train_size = 0
        if self.train_data is not None:
            train_size = len(self.train_data['y'])

        test_size = 0
        if self.eval_data is not None:
            test_size = len(self.eval_data['y'])
        return train_size + test_size

    def save_model(self, checkpoints_folder='checkpoints'):
        path = '{}/{}/worker_model.ckpt'.format(checkpoints_folder, self.id)
        return torch.save(self.model.cur_model, path)

    def move_model_to_device(self):
        self.model.move_model_to_device(self.processing_device)
