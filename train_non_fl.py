import math
import os
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


class Trainer:
    def __init__(self, dataset='femnist', lr=0.01):
        assert dataset in ['femnist', 'sent140']
        self.dataset = dataset
        self.model = CnnFemnist() if dataset == 'femnist' else CnnSent()
        self.lr = lr
        self._optimizer = optim.SGD(params=self.model.parameters(),
                                    lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.read_data_from_dir()

    def read_data_from_dir(self):
        print('Reading data. This may take a while...')
        data_dir = os.path.join('data', self.dataset, 'data', 'all_data')
        try:
            self.train_data, self.test_data = Trainer.read_data(data_dir)
        except FileNotFoundError:
            _ = subprocess.call("./ data_to_json.sh",
                                cwd=os.path.join('data', self.dataset, 'preprocess'),
                                shell=True)
            self.train_data, self.test_data = Trainer.read_data(data_dir)

    @staticmethod
    def read_data(data_dir):
        data = {'x': [], 'y': []}
        files = os.listdir(data_dir)
        files = [f for f in files if f.endswith('.json')]
        for i, f in enumerate(files):
            print('Reading file {} out of {}'.format(i, len(files)), end='\r')
            file_path = os.path.join(data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
                users = [user for user, _ in cdata['user_data'].items()]
                for user in users:
                    data['x'] += cdata['user_data'][user]['x']
                    data['y'] += cdata['user_data'][user]['y']
        print()
        print('Splitting train and test files...')
        X_train, X_test, y_train, y_test = train_test_split(data['x'],
                                                            data['y'],
                                                            test_size=0.33,
                                                            random_state=42)
        print('Gathering xs and ys on single dictionary...')
        train_data = {'x': X_train, 'y': y_train}
        test_data = {'x': X_test, 'y': y_test}
        return train_data, test_data

    def train(self, epochs=100, batch_size=10):
        print('--- Start training ---')
        for epoch in range(epochs):
            # Simulate server model training on selected clients' data
            final_loss, _, _, _ = self.train_single_epoch(epoch=epoch,
                                                          epochs=epochs,
                                                          batch_size=batch_size)

            # Test model
            _ = self.test_model(batch_size=batch_size)
        print('--- Training finished! ---')
        # Save server model
        ckpt_path = os.path.join('checkpoints', self.dataset)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        save_path = self.save_model(checkpoints_folder=ckpt_path)
        print('Model saved in path: %s' % save_path)

    def train_single_epoch(self, epoch, epochs, batch_size=10):
        start = time.time()
        self.model.train()
        running_loss = 0.
        last_loss = 0.
        counter = 0.
        predictions = []
        labels_list = []
        for batch_input, batch_label in batch_data(self.train_data, batch_size):
            batch_input, batch_label = self.preprocess_input_output(batch_input, batch_label)
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
            metrics = {'acc': accuracy_score(np.asarray(labels_list), np.asarray(predictions)),
                       'f1': f1_score(np.asarray(labels_list), np.asarray(predictions), average='weighted')}
            self.print_message(epoch=epoch,
                               epochs=epochs,
                               index_train_batch=counter,
                               batch_size=batch_size,
                               train_loss=running_loss / counter,
                               train_mets=metrics)
        print()
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
                self.print_test_message(index_batch=counter,
                                        batch_size=batch_size,
                                        metrics=metrics)
            print()
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
            inputs = inputs.type(torch.FloatTensor)
            labels = torch.from_numpy(np.array(batch_output))
            return inputs, labels
        elif self.dataset == 'sent140':
            try:
                _, indd, _ = get_word_emb_arr('enea_fl/models/embs.json')
            except FileNotFoundError:
                _ = subprocess.call("./enea_fl/models/get_embs.sh", shell=True)
                _, indd, _ = get_word_emb_arr('enea_fl/models/embs.json')
            x_batch = [e[4] for e in batch_input]
            x_batch = [line_to_indices(e, indd, max_words=SentConfig().max_sen_len) for e in x_batch]
            inputs = torch.from_numpy(np.array(x_batch))
            labels = torch.from_numpy(np.array(batch_output))
            return inputs, labels
        else:
            raise ValueError('Dataset "{}" is not available!'.format(self.dataset))

    def print_message(self, epoch, epochs, index_train_batch, batch_size, train_loss, train_mets):
        message = '| Epoch: {}/{} | LR: {:.5f} |'.format(epoch + 1,
                                                         epochs,
                                                         self.lr)
        bar_length = 10
        total_train_batches = math.ceil(len(self.train_data['x']) / batch_size)
        progress = float(index_train_batch) / float(total_train_batches)
        if progress >= 1.:
            progress = 1
        block = int(round(bar_length * progress))
        message += '[{}]'.format('=' * block + ' ' * (bar_length - block))
        message += '| TRAIN: loss={:.5f} '.format(train_loss)
        if train_mets is not None:
            train_metrics_message = ''
            for metric_name, metric_value in train_mets.items():
                train_metrics_message += '{}={:.5f} '.format(metric_name,
                                                             metric_value)
            message += train_metrics_message
        message += '|'
        # message += 'Loss weights are: {}'.format(self.criterion_reg.weight.numpy())
        print(message, end='\r')

    def print_test_message(self, index_batch, batch_size, metrics):
        message = '| '
        bar_length = 10
        total_batches = math.ceil(len(self.test_data['x']) / batch_size)
        progress = float(index_batch) / float(total_batches)
        if progress >= 1.:
            progress = 1
        block = int(round(bar_length * progress))
        message += '[{}] | TEST: '.format('=' * block + ' ' * (bar_length - block))
        if metrics is not None:
            metrics_message = ''
            for metric_name, metric_value in metrics.items():
                metrics_message += '{}={:.5f} '.format(metric_name,
                                                       metric_value)
            message += metrics_message
        message += '|'
        print(message, end='\r')

    def save_model(self, checkpoints_folder='checkpoints'):
        # Save server model
        path = '{}/server_model.ckpt'.format(checkpoints_folder)
        torch.save(self.model, path)


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
