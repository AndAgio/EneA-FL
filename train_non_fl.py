import math
import os
import shutil
import time
import json
import argparse
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from enea_fl.models.utils import batch_data, line_to_indices, get_word_emb_arr
from enea_fl.models import CnnSent, CnnFemnist, SentConfig
from enea_fl.utils import get_logger, get_free_gpu


class Trainer:
    def __init__(self, dataset='femnist', lr=0.01):
        assert dataset in ['femnist', 'sent140']
        self.dataset = dataset
        self.clean_previous_logger()
        self.logger = get_logger(node_type='non_fl', node_id='0', log_folder=os.path.join('logs', dataset))
        self.model = CnnFemnist() if dataset == 'femnist' else CnnSent()
        self.processing_device = torch.device('cuda:{}'.format(get_free_gpu()) if torch.cuda.is_available()
                                              else 'cpu')
        self.logger.print_it('Using a {} for training and inference!'.format(self.processing_device))
        self.model = self.model.to(self.processing_device)
        self.lr = lr
        self._optimizer = optim.SGD(params=self.model.parameters(),
                                    lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.read_data_from_dir()

    def read_data_from_dir(self):
        self.logger.print_it('Reading data. This may take a while...')
        data_dir = os.path.join('data', self.dataset, 'data', 'all_data')
        try:
            self.train_data, self.test_data = self.read_data(data_dir)
        except FileNotFoundError:
            _ = subprocess.call("./ data_to_json.sh",
                                cwd=os.path.join('data', self.dataset, 'preprocess'),
                                shell=True)
            self.train_data, self.test_data = self.read_data(data_dir)

    def read_data(self, data_dir):
        data = {'x': [], 'y': []}
        files = os.listdir(data_dir)
        files = [f for f in files if f.endswith('.json')]
        for i, f in enumerate(files):
            self.logger.print_it_same_line('Reading file {} out of {}'.format(i+1, len(files)))
            file_path = os.path.join(data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
                users = [user for user, _ in cdata['user_data'].items()]
                for user in users:
                    data['x'] += cdata['user_data'][user]['x']
                    data['y'] += cdata['user_data'][user]['y']
        self.logger.set_logger_newline()
        self.logger.print_it('Splitting train and test files...')
        x_train, x_test, y_train, y_test = train_test_split(data['x'],
                                                            data['y'],
                                                            test_size=0.33,
                                                            random_state=42)
        self.logger.print_it('Gathering xs and ys on single dictionary...')
        train_data = {'x': x_train, 'y': y_train}
        test_data = {'x': x_test, 'y': y_test}
        return train_data, test_data

    def train(self, epochs=100, batch_size=10):
        self.logger.print_it('--- Start training ---')
        for epoch in range(epochs):
            self.logger.print_it('===== | Epoch: {}/{} | LR = {:.5f} | BATCH = {} | ======='.format(epoch + 1, epochs,
                                                                                     self.lr, batch_size))
            # Simulate server model training on selected clients' data
            final_loss, _, _, _ = self.train_single_epoch(batch_size=batch_size)
            # Test model
            _ = self.test_model(batch_size=batch_size)
            self.logger.print_it('============================================')
        self.logger.print_it('--- Training finished! ---')
        # Save server model
        ckpt_path = os.path.join('checkpoints', self.dataset)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        save_path = self.save_model(checkpoints_folder=ckpt_path)
        self.logger.print_it('Model saved in path: %s' % save_path)

    def train_single_epoch(self, batch_size=10):
        start = time.time()
        self.model.train()
        running_loss = 0.
        last_loss = 0.
        counter = 0.
        predictions = []
        labels_list = []
        for batch_input, batch_label in batch_data(self.train_data, batch_size):
            batch_input, batch_label = self.preprocess_input_output(batch_input, batch_label)
            self.logger.print_it('Model device:')
            for i in self.model.named_parameters():
                self.logger.print_it(f"{i[0]} -> {i[1].device}")
            self.logger.print_it('Input device: {}'.format(batch_input.get_device()))
            self.logger.print_it('Label device: {}'.format(batch_label.get_device()))
            self._optimizer.zero_grad()
            outputs = self.model(batch_input)
            loss = self.criterion(outputs, batch_label)
            loss.backward()
            self._optimizer.step()
            running_loss += loss.item()
            pred_labels = torch.argmax(outputs, dim=1)
            predictions += pred_labels.detach().cpu().numpy().tolist()
            labels_list += batch_label.detach().cpu().numpy().tolist()
            counter += 1
            metrics = {'loss': running_loss / counter,
                       'acc': accuracy_score(np.asarray(labels_list), np.asarray(predictions)),
                       'f1': f1_score(np.asarray(labels_list), np.asarray(predictions), average='weighted')}
            self.print_message(index_batch=counter,
                               batch_size=batch_size,
                               metrics=metrics,
                               mode='train')
        self.logger.set_logger_newline()
        final_loss = running_loss / counter
        stop = time.time()
        energy = 0.  # TODO: find how to compute energy here
        comp = 0.  # TODO: find how to compute flops of model
        return final_loss, energy, stop - start, comp

    def test_model(self, batch_size=10):
        predictions = []
        labels_list = []
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.
            counter = 0
            for batch_input, batch_label in batch_data(self.test_data, batch_size):
                batch_input, batch_label = self.preprocess_input_output(batch_input, batch_label)
                output = self.model(batch_input)
                running_loss += self.criterion(output, batch_label).item()
                # preds_softmax = torch.nn.functional.softmax(output, dim=1)
                pred_labels = torch.argmax(output, dim=1)
                predictions += pred_labels.detach().cpu().numpy().tolist()
                labels_list += batch_label.detach().cpu().numpy().tolist()
                counter += 1
                metrics = {'acc': accuracy_score(np.asarray(labels_list), np.asarray(predictions)),
                           'f1': f1_score(np.asarray(labels_list), np.asarray(predictions), average='weighted')}
                self.print_message(index_batch=counter,
                                   batch_size=batch_size,
                                   metrics=metrics,
                                   mode='test')
            self.logger.set_logger_newline()
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

    def print_message(self, index_batch, batch_size, metrics, mode='train'):
        message = '|'
        bar_length = 10
        total_samples = len(self.train_data['x']) if mode == 'train' else len(
            self.test_data) if mode == 'test' else len(self.val_data)
        total_batches = math.ceil(total_samples / batch_size)
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

    def save_model(self, checkpoints_folder='checkpoints'):
        # Save server model
        path = '{}/server_model.ckpt'.format(checkpoints_folder)
        torch.save(self.model, path)
        return path

    def clean_previous_logger(self):
        log_folder = os.path.join('logs', self.dataset, 'non_fl')
        shutil.rmtree(log_folder)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='name of dataset;', type=str, choices=['sent140', 'femnist'], required=True)
    parser.add_argument('--epochs', help='number of epochs;', type=int, default=100)
    parser.add_argument('--batch_size', help='batch size when clients train on data;', type=int, default=10)
    parser.add_argument('--lr', help='learning rate for local optimizers;', type=float, default=-1, required=False)
    return parser.parse_args()


def main():
    args = parse_args()
    my_trainer = Trainer(dataset=args.dataset,
                         lr=args.lr)
    my_trainer.train(epochs=args.epochs,
                     batch_size=args.batch_size)
    my_trainer.save_model()


if __name__ == '__main__':
    main()
