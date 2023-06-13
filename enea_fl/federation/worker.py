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
        self.energy_policy = energy_policy
        # print('worker_id: {} -> device={}, policy={}, model={}'.format(worker_id,
        #                                                                device_type,
        #                                                                energy_policy,
        #                                                                self._model))
        self.train_data = train_data
        self.eval_data = eval_data

    def train(self, batch_size=10):
        """Trains on self.model using the client's train_data.

        Args:
            batch_size: Size of training batches.
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            update: set of weights
            update_size: number of bytes in update
        """
        train_steps = self.compute_local_energy_policy(batch_size=batch_size)
        energy_used, time_taken, comp = self.model.train(self.train_data, train_steps, batch_size)
        num_train_samples = train_steps * batch_size
        return energy_used, time_taken, comp, num_train_samples

    def test(self, set_to_use='test'):
        """Tests self.model on self.test_data.

        Args:
            set_to_use. Set to test on. Should be in ['train', 'test'].
        Return:
            dict of metrics returned by the model.
        """
        assert set_to_use in ['train', 'test', 'val']
        if set_to_use == 'train':
            data = self.train_data
        elif set_to_use == 'test' or set_to_use == 'val':
            data = self.eval_data
        else:
            raise ValueError('Something worng with data in testing!')
        return self.model.test_my_model(data)

    def compute_local_energy_policy(self, batch_size=10):
        if self.energy_policy == 'normal':
            return math.ceil(self.num_train_samples / batch_size)
        elif self.energy_policy == 'conservative':
            return math.floor((self.num_train_samples/batch_size)/10)
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
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        if self.eval_data is None:
            return 0
        return len(self.eval_data['y'])

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        if self.train_data is None:
            return 0
        return len(self.train_data['y'])

    @property
    def num_samples(self):
        """Number samples for this client.

        Return:
            int: Number of samples for this client
        """
        train_size = 0
        if self.train_data is not None:
            train_size = len(self.train_data['y'])

        test_size = 0
        if self.eval_data is not None:
            test_size = len(self.eval_data['y'])
        return train_size + test_size

    @property
    def model(self):
        """Returns this client reference to model being trained"""
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model = model

    def save_model(self, checkpoints_folder='checkpoints'):
        path = '{}/{}/worker_model.ckpt'.format(checkpoints_folder, self.id)
        return torch.save(self.model.cur_model, path)
