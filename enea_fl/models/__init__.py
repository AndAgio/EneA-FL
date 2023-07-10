import math

from .femnist import CnnFemnist
from .sent140 import CnnSent
from .utils import read_data, batch_data, get_word_emb_arr, line_to_indices
from .config_files import SentConfig

import time
import subprocess
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from enea_fl.utils import DumbLogger, get_free_gpu


class WorkerModel:
    def __init__(self, dataset='femnist', indexization=None, lr=0.01):
        assert dataset in ['femnist', 'sent140']
        if dataset == 'sent140' and indexization is None:
            raise ValueError('Indexization should be a valid input when'
                             ' constructing WorkerModel objects for the Sent140 task!')
        self.dataset = dataset
        self.model = CnnFemnist() if dataset == 'femnist' else CnnSent()
        self.lr = lr
        self._optimizer = optim.SGD(params=self.model.parameters(),
                                    lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.processing_device = torch.device('cuda:0'  # {}'.format(get_free_gpu())
                                              if torch.cuda.is_available() else 'cpu')
        self.logger = DumbLogger
        self.indexization = indexization

    @property
    def size(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @property
    def cur_model(self):
        return self.model

    def set_logger(self, logger):
        self.logger = logger

    def set_weights(self, aggregated_numpy):
        for i, param in enumerate(self.model.parameters()):
            param.data = torch.from_numpy(aggregated_numpy[i]).type('torch.FloatTensor')

    def get_weights(self):
        return np.array([param.detach().cpu().numpy() for param in self.model.parameters()],
                        dtype=object)

    def train(self, train_data, train_steps=100, batch_size=10, lr=0.1):
        self._optimizer.param_groups[0]['lr'] = lr
        self.model.to(self.processing_device)
        self.model.train()
        predictions = []
        labels_list = []
        metrics = {}
        running_loss = 0.
        counter = 0
        for batch_input, batch_label in batch_data(train_data, batch_size):
            batch_input, batch_label = self.preprocess_input_output(batch_input, batch_label)
            self._optimizer.zero_grad()
            outputs = self.model(batch_input)
            loss = self.criterion(outputs, batch_label)
            loss.backward()
            self._optimizer.step()
            running_loss += loss.item()
            last_loss = loss.item()
            counter += 1
            pred_labels = torch.argmax(outputs, dim=1)
            predictions += pred_labels.detach().cpu().numpy().tolist()
            labels_list += batch_label.detach().cpu().numpy().tolist()
            metrics = {'loss': running_loss / counter,
                       'acc': accuracy_score(np.asarray(labels_list), np.asarray(predictions)),
                       'f1': f1_score(np.asarray(labels_list), np.asarray(predictions), average='weighted')}
            self.print_message(index_batch=counter, total_batches=train_steps, metrics=metrics, mode='train')
            if counter >= train_steps:
                break
        final_loss = running_loss / counter
        return metrics

    def test_my_model(self, test_data, batch_size=10):
        self.model.to(self.processing_device)
        self.model.eval()
        with torch.no_grad():
            predictions = []
            labels_list = []
            counter = 0.
            for batch_input, batch_label in batch_data(test_data, batch_size):
                batch_input, batch_label = self.preprocess_input_output(batch_input, batch_label)
                outputs = self.model(batch_input)
                counter += 1.
                pred_labels = torch.argmax(outputs, dim=1)
                predictions += pred_labels.detach().cpu().numpy().tolist()
                labels_list += batch_label.detach().cpu().numpy().tolist()
                metrics = {'acc': accuracy_score(np.asarray(labels_list), np.asarray(predictions)),
                           'f1': f1_score(np.asarray(labels_list), np.asarray(predictions), average='weighted')}
                self.print_message(index_batch=counter, total_batches=math.ceil(len(test_data['y']) / batch_size),
                                   metrics=metrics, mode='test local')
            return metrics

    def test_other_model(self, test_data, ids, other_model, results, batch_size=10):
        other_model.to(self.processing_device)
        other_model.eval()
        with torch.no_grad():
            predictions = []
            labels_list = []
            counter = 0
            for batch_input, batch_label in batch_data(test_data, batch_size):
                batch_input, batch_label = self.preprocess_input_output(batch_input, batch_label)
                outputs = other_model(batch_input)
                counter += 1
                pred_labels = torch.argmax(outputs, dim=1)
                predictions += pred_labels.detach().cpu().numpy().tolist()
                labels_list += batch_label.detach().cpu().numpy().tolist()
                metrics = {'acc': accuracy_score(np.asarray(labels_list), np.asarray(predictions)),
                           'f1': f1_score(np.asarray(labels_list), np.asarray(predictions), average='weighted')}
                self.print_message(index_batch=counter, total_batches=math.ceil(len(test_data['y']) / batch_size),
                                   metrics=metrics, mode='test other')
            results[ids] = metrics
        return results

    def test_final_model(self, final_model, test_data, batch_size=10):
        predictions = []
        labels_list = []
        final_model.to(self.processing_device)
        final_model.eval()
        with torch.no_grad():
            counter = 0
            for batch_input, batch_label in batch_data(test_data, batch_size):
                batch_input, batch_label = self.preprocess_input_output(batch_input, batch_label)
                output = final_model(batch_input)
                preds_softmax = torch.nn.functional.softmax(output, dim=1)
                pred_labels = torch.argmax(preds_softmax, dim=1)
                predictions += pred_labels.detach().cpu().numpy().tolist()
                labels_list += batch_label.detach().cpu().numpy().tolist()
                counter += 1
                metrics = {'acc': accuracy_score(np.asarray(labels_list), np.asarray(predictions)),
                           'f1': f1_score(np.asarray(labels_list), np.asarray(predictions), average='weighted')}
                self.print_message(index_batch=counter, total_batches=math.ceil(len(test_data['y']) / batch_size),
                                   metrics=metrics, mode='test global')
            # Compute accuracy
            f1 = f1_score(np.asarray(labels_list), np.asarray(predictions), average='weighted')
            accuracy = accuracy_score(np.asarray(labels_list), np.asarray(predictions))
        return {
            'accuracy': accuracy,
            'f1': f1
        }

    def preprocess_input_output(self, batch_input, batch_output):
        if self.dataset == 'femnist':
            inputs = torch.from_numpy(np.array(batch_input).reshape((len(batch_input), 1, 28, 28)))
            inputs = inputs.type(torch.FloatTensor).to(self.processing_device)
            labels = torch.from_numpy(np.array(batch_output)).to(self.processing_device)
            return inputs, labels
        elif self.dataset == 'sent140':
            x_batch = [e[4] for e in batch_input]
            x_batch = [line_to_indices(e, self.indexization, max_words=SentConfig().max_sen_len) for e in x_batch]
            inputs = torch.from_numpy(np.array(x_batch)).type(torch.LongTensor).permute(1, 0).to(self.processing_device)
            labels = torch.from_numpy(np.array(batch_output)).to(self.processing_device)
            return inputs, labels
        else:
            raise ValueError('Dataset "{}" is not available!'.format(self.dataset))

    def move_model_to_device(self, processing_device):
        self.processing_device = processing_device
        self.model = self.model.to(self.processing_device)

    def print_message(self, index_batch, total_batches, metrics, mode='train'):
        message = '|'
        bar_length = 10
        progress = float(index_batch) / float(total_batches)
        if progress >= 1.:
            progress = 1
        block = int(round(bar_length * progress))
        message += '[{}]'.format('=' * block + ' ' * (bar_length - block))
        message += '| {}: '.format(mode.upper())
        if metrics is not None:
            train_metrics_message = ''
            index = 0
            for metric_name, metric_value in metrics.items():
                train_metrics_message += '{}={:.5f}{} '.format(metric_name,
                                                               metric_value,
                                                               ',' if index < len(metrics.keys()) - 1 else '')
                index += 1
            message += train_metrics_message
        message += '|'.ljust(60)
        self.logger.print_it_same_line(message)


class ServerModel:
    def __init__(self, dataset='femnist', indexization=None):
        assert dataset in ['femnist', 'sent140']
        if dataset == 'sent140' and indexization is None:
            raise ValueError('Indexization should be a valid input when'
                             ' constructing WorkerModel objects for the Sent140 task!')
        self.dataset = dataset
        self.indexization = indexization
        self.model = CnnFemnist() if dataset == 'femnist' else CnnSent()
        self.processing_device = torch.device('cuda:0'  # {}'.format(get_free_gpu())
                                              if torch.cuda.is_available() else 'cpu')

    @property
    def size(self):
        return self.model.size

    @property
    def cur_model(self):
        return self.model

    def send_to(self, workers):
        for w in workers:
            w.set_weights(np.array([param.detach().cpu().numpy() for param in self.model.parameters()],
                                   dtype=object))

    def set_weights(self, aggregated_numpy):
        for i, param in enumerate(self.model.parameters()):
            param.data = torch.from_numpy(aggregated_numpy[i]).type('torch.FloatTensor')

    def get_weights(self):
        return np.array([param.detach().cpu().numpy() for param in self.model.parameters()],
                        dtype=object)

    def move_model_to_device(self, processing_device):
        self.processing_device = processing_device
        self.model = self.model.to(self.processing_device)

    def test(self, test_data, batch_size=10):
        predictions = []
        labels_list = []
        self.model.to(self.processing_device)
        self.model.eval()
        with torch.no_grad():
            for batch_input, batch_label in batch_data(test_data, batch_size):
                batch_input, batch_label = self.preprocess_input_output(batch_input, batch_label)
                output = self.model(batch_input)
                preds_softmax = torch.nn.functional.softmax(output, dim=1)
                pred_labels = torch.argmax(preds_softmax, dim=1)
                predictions += pred_labels.detach().cpu().numpy().tolist()
                labels_list += batch_label.detach().cpu().numpy().tolist()
            # Compute accuracy
            f1 = f1_score(np.asarray(labels_list), np.asarray(predictions), average='weighted')
            accuracy = accuracy_score(np.asarray(labels_list), np.asarray(predictions))
        return {'accuracy': accuracy, 'f1': f1}

    def preprocess_input_output(self, batch_input, batch_output):
        if self.dataset == 'femnist':
            inputs = torch.from_numpy(np.array(batch_input).reshape((len(batch_input), 1, 28, 28)))
            inputs = inputs.type(torch.FloatTensor).to(self.processing_device)
            labels = torch.from_numpy(np.array(batch_output)).to(self.processing_device)
            return inputs, labels
        elif self.dataset == 'sent140':
            x_batch = [e[4] for e in batch_input]
            x_batch = [line_to_indices(e, self.indexization, max_words=SentConfig().max_sen_len) for e in x_batch]
            inputs = torch.from_numpy(np.array(x_batch)).type(torch.LongTensor).permute(1, 0).to(self.processing_device)
            labels = torch.from_numpy(np.array(batch_output)).to(self.processing_device)
            return inputs, labels
        else:
            raise ValueError('Dataset "{}" is not available!'.format(self.dataset))

    def create_copy(self):
        return ServerModel(self.dataset, self.indexization)
