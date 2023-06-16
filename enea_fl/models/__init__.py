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
from enea_fl.utils import DumbLogger


class WorkerModel:
    def __init__(self, dataset='femnist', lr=0.01):
        assert dataset in ['femnist', 'sent140']
        self.dataset = dataset
        self.model = CnnFemnist() if dataset == 'femnist' else CnnSent()
        self.lr = lr
        self._optimizer = optim.SGD(params=self.model.parameters(),
                                    lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.processing_device = 'cpu'
        self.logger = DumbLogger

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

    def train(self, train_data, train_steps=100, batch_size=10):
        start = time.time()
        self.model.train()
        predictions = []
        labels_list = []
        running_loss = 0.
        last_loss = 0.
        counter = 0
        for batch_input, batch_label in batch_data(train_data, batch_size):
            batch_input, batch_label = self.preprocess_input_output(batch_input, batch_label)
            self._optimizer.zero_grad()
            outputs = self.model(batch_input)
            loss = self.criterion(outputs, batch_label)
            loss.backward()
            self._optimizer.step()
            running_loss += loss.item()
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
        stop = time.time()
        energy = 0.  # TODO: find how to compute energy here
        comp = 0.  # TODO: find how to compute flops of model
        return energy, stop - start, comp

    def test_my_model(self, test_data, batch_size=10):
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.
            predictions = []
            labels_list = []
            counter = 0.
            for batch_input, batch_label in batch_data(test_data, batch_size):
                batch_input, batch_label = self.preprocess_input_output(batch_input, batch_label)
                outputs = self.model(batch_input)
                running_loss += self.criterion(outputs, batch_label).item()
                counter += 1.
                pred_labels = torch.argmax(outputs, dim=1)
                predictions += pred_labels.detach().cpu().numpy().tolist()
                labels_list += batch_label.detach().cpu().numpy().tolist()
                metrics = {'loss': running_loss / counter,
                           'acc': accuracy_score(np.asarray(labels_list), np.asarray(predictions)),
                           'f1': f1_score(np.asarray(labels_list), np.asarray(predictions), average='weighted')}
                self.print_message(index_batch=counter, total_batches=math.ceil(len(test_data['y']) / batch_size),
                                   metrics=metrics, mode='test local')
            return {'loss': running_loss / counter}

    def test_other_model(self, test_data, ids, other_model, results, batch_size=10):
        other_model.eval()
        with torch.no_grad():
            running_loss = 0.
            predictions = []
            labels_list = []
            counter = 0
            for batch_input, batch_label in batch_data(test_data, batch_size):
                batch_input, batch_label = self.preprocess_input_output(batch_input, batch_label)
                outputs = other_model(batch_input)
                running_loss += self.criterion(outputs, batch_label).item()
                counter += 1
                pred_labels = torch.argmax(outputs, dim=1)
                predictions += pred_labels.detach().cpu().numpy().tolist()
                labels_list += batch_label.detach().cpu().numpy().tolist()
                metrics = {'loss': running_loss / counter,
                           'acc': accuracy_score(np.asarray(labels_list), np.asarray(predictions)),
                           'f1': f1_score(np.asarray(labels_list), np.asarray(predictions), average='weighted')}
                self.print_message(index_batch=counter, total_batches=math.ceil(len(test_data['y']) / batch_size),
                                   metrics=metrics, mode='test other')
            results[ids] = metrics
        return results

    def test_final_model(self, final_model, test_data, batch_size=10):
        predictions = []
        labels_list = []
        final_model.eval()
        with torch.no_grad():
            running_loss = 0.
            counter = 0
            for batch_input, batch_label in batch_data(test_data, batch_size):
                batch_input, batch_label = self.preprocess_input_output(batch_input, batch_label)
                output = final_model(batch_input)
                running_loss += self.criterion(output, batch_label).item()
                preds_softmax = torch.nn.functional.softmax(output, dim=1)
                pred_labels = torch.argmax(preds_softmax, dim=1)
                predictions += pred_labels.detach().cpu().numpy().tolist()
                labels_list += batch_label.detach().cpu().numpy().tolist()
                counter += 1
                metrics = {'loss': running_loss / counter,
                           'acc': accuracy_score(np.asarray(labels_list), np.asarray(predictions)),
                           'f1': f1_score(np.asarray(labels_list), np.asarray(predictions), average='weighted')}
                self.print_message(index_batch=counter, total_batches=math.ceil(len(test_data['y']) / batch_size),
                                   metrics=metrics, mode='test global')
            # Compute accuracy
            f1 = f1_score(np.asarray(labels_list), np.asarray(predictions), average='weighted')
            accuracy = accuracy_score(np.asarray(labels_list), np.asarray(predictions))
        return {
            'loss': running_loss / counter,
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
            try:
                _, indd, _ = get_word_emb_arr('enea_fl/models/embs.json')
            except FileNotFoundError:
                _ = subprocess.call("./enea_fl/models/get_embs.sh", shell=True)
                _, indd, _ = get_word_emb_arr('enea_fl/models/embs.json')
            x_batch = [e[4] for e in batch_input]
            x_batch = [line_to_indices(e, indd, max_words=SentConfig().max_sen_len) for e in x_batch]
            inputs = torch.from_numpy(np.array(x_batch)).to(self.processing_device)
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
        message += '|'
        self.logger.print_it_same_line(message)


class ServerModel:
    def __init__(self, dataset='femnist'):
        assert dataset in ['femnist', 'sent140']
        self.dataset = dataset
        self.model = CnnFemnist() if dataset == 'femnist' else CnnSent()
        self.processing_device = 'cpu'

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
