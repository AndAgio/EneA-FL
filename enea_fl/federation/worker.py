import math
import warnings
import torch


class Worker:
    def __init__(self,
                 worker_id,
                 device_type='raspberry',
                 energy_policy='normal',
                 train_data={'x': [], 'y': []},
                 eval_data={'x': [], 'y': []},
                 model=None):
        self._model = model
        self.id = worker_id
        self.device_type = device_type
        gpu = True if device_type in ['nano', 'jetson'] else False
        self.processing_device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
        print('Worker {} is running on a {} and using a {} for training and inference!'.format(self.id,
                                                                                               self.device_type,
                                                                                               self.processing_device))
        self.move_model_to_device()
        self.energy_policy = energy_policy
        self.train_data = train_data
        self.eval_data = eval_data

    def train(self, batch_size=10):
        train_steps = self.compute_local_energy_policy(batch_size=batch_size)
        energy_used, time_taken, comp = self.model.train(train_data=self.train_data,
                                                         train_steps=train_steps,
                                                         batch_size=batch_size)
        num_train_samples = train_steps * batch_size
        return energy_used, time_taken, comp, num_train_samples

    def test_local(self, set_to_use='test'):
        data = self.select_data_for_testing(set_to_use)
        return self.model.test_my_model(test_data=data)

    def test_global(self, model_to_test, set_to_use='test'):
        data = self.select_data_for_testing(set_to_use)
        return self.model.test_final_model(model_to_test=model_to_test,
                                           test_data=data)

    def select_data_for_testing(self, set_to_use='test'):
        assert set_to_use in ['train', 'test', 'val']
        if set_to_use == 'train':
            data = self.train_data
        elif set_to_use == 'test' or set_to_use == 'val':
            data = self.eval_data
        else:
            raise ValueError('Something wrong with data in testing!')
        return data

    def compute_local_energy_policy(self, batch_size=10):
        if self.energy_policy == 'normal':
            return math.ceil(self.num_train_samples / batch_size)
        elif self.energy_policy == 'conservative':
            return math.floor((self.num_train_samples / batch_size) / 10)
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

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model = model

    def save_model(self, checkpoints_folder='checkpoints'):
        path = '{}/{}/worker_model.ckpt'.format(checkpoints_folder, self.id)
        return torch.save(self.model.cur_model, path)

    def move_model_to_device(self):
        self.model.move_model_to_device(self.processing_device)
