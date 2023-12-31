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
from enea_fl.models import SentModel, FemnistModel, MnistModel, SentConfig
from enea_fl.utils import get_logger, get_free_gpu


class Trainer:
    def __init__(self, dataset='femnist', lr=0.01, batch_size=10, is_iot=True):
        assert dataset in ['femnist', 'mnist', 'sent140']
        self.logger = get_logger(node_type='non_fl', node_id='0', log_folder=os.path.join('logs', dataset))
        self.logger.print_it('Istantiating a Trainer object for {} dataset!'.format(dataset))
        self.dataset = dataset
        self.batch_size = batch_size
        self.is_iot = is_iot
        # self.logger.print_it('Cleaning previous logger!')
        # self.clean_previous_logger()
        self.logger.print_it('Istantiating a model!')
        if dataset == 'femnist':
            self.model = FemnistModel()
        elif dataset == 'mnist':
            self.model = MnistModel()
        else:
            self.model = SentModel()
        self.logger.print_it('Model: {}'.format(self.model))
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
        if self.dataset == 'sent140':
            self.indexization = self.gather_indexization()
        else:
            self.indexization = None

        self.tot_flops = self.compute_total_number_of_flops()

    def compute_total_number_of_flops(self):
        total_flops = 0
        for name, layer in self.model.named_children():
            if isinstance(layer, nn.Linear):
                total_flops += layer.in_features * layer.out_features * self.batch_size
            elif isinstance(layer, nn.Conv1d):
                total_flops += layer.in_channels * layer.out_channels * layer.kernel_size[0] * self.batch_size
            elif isinstance(layer, nn.Conv2d):
                total_flops += 2 * layer.in_channels * layer.out_channels * layer.kernel_size[0] * self.batch_size

        return total_flops

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
        if self.is_iot:
            self.logger.print_it('[iot mode] Using less data for training!')
            if self.dataset == 'sent140':
                files = [f for f in files if 'small' in f]
            else:
                files = files[:1]
        for i, f in enumerate(files):
            self.logger.print_it_same_line('Reading file {} out of {}'.format(i + 1, len(files)))
            file_path = os.path.join(data_dir, f)
            self.logger.print_it("file_path: {}".format(file_path))
            with open(file_path, 'r', encoding="utf-8") as inf:
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
        self.logger.print_it(' Start training '.center(60, '-'))
        for epoch in range(epochs):
            self.logger.print_it(' | Epoch: {}/{} | LR = {:.5f} | BATCH = {} | '.format(epoch + 1,
                                                                                        epochs,
                                                                                        self.lr,
                                                                                        batch_size).center(60, '-'))
            # Simulate server model training on selected clients' data
            try:
                final_loss, _, _, _ = self.train_single_epoch(batch_size=batch_size)
            except Exception as e:
                self.logger.print_it('Error during training: {}'.format(e))
                continue
            # Test model
            _ = self.test_model(batch_size=batch_size)
            self.logger.print_it(''.center(60, '-'))
        self.logger.print_it(' Training finished! '.center(60, '-'))
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
        print("sum(1 for _ in gen)", sum(1 for _ in batch_data(self.train_data, batch_size)))
        for batch_input, batch_label in batch_data(self.train_data, batch_size):
            batch_input, batch_label = self.preprocess_input_output(batch_input, batch_label)
            self._optimizer.zero_grad()
            outputs = self.model(batch_input)
            # parse batch_label to long (raspberry pi)
            batch_label = batch_label.long()
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
            x_batch = [e[4] for e in batch_input]
            x_batch = [line_to_indices(e, self.indexization, max_words=SentConfig().max_sen_len) for e in x_batch]
            inputs = torch.from_numpy(np.array(x_batch)).type(torch.LongTensor).permute(1, 0).to(self.processing_device)
            labels = torch.from_numpy(np.array(batch_output)).to(self.processing_device)
            return inputs, labels
        else:
            raise ValueError('Dataset "{}" is not available!'.format(self.dataset))

    def gather_indexization(self):
        self.logger.print_it('Reading word indexization from GloVe\'s json...')
        try:
            # _, indd, _ = get_word_emb_arr('enea_fl/models/embs.json')
            # _, _, indd, _ = get_word_emb_arr(self.model.embs)
            _, _, indd, _ = get_word_emb_arr('enea_fl/models/embs.json')
        except Exception as e:
            print(e)
            # _ = subprocess.call("./enea_fl/models/get_embs.sh", shell=True)
            # _, _, indd, _ = get_word_emb_arr(self.model.embs)
            _, _, indd, _ = get_word_emb_arr('enea_fl/models/embs.json')
        return indd

    def print_message(self, index_batch, batch_size, metrics, mode='train'):
        message = '|'
        bar_length = 10
        total_samples = len(self.train_data['x']) if mode == 'train' else len(
            self.test_data['x']) if mode == 'test' else len(self.val_data['x'])
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
        message += '|'.ljust(60)
        self.logger.print_it_same_line(message)

    def save_model(self, checkpoints_folder='checkpoints'):
        # Save server model
        path = '{}/server_model.ckpt'.format(checkpoints_folder)
        torch.save(self.model, path)
        return path

    def clean_previous_logger(self):
        log_folder = os.path.join('logs', self.dataset, 'non_fl')
        if os.path.exists(log_folder):
            shutil.rmtree(log_folder)

    def warm_up_model(self):
        self.logger.print_it('Warming up model...')
        self.model.train()
        i = 0
        for batch_input, batch_label in batch_data(self.train_data, 10):
            if i == 100:
                break
            batch_input, batch_label = self.preprocess_input_output(batch_input, batch_label)
            self.model(batch_input)
            i += 1
        self.logger.print_it('Model warmed up!')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='name of dataset;', type=str, choices=['sent140', 'femnist'], required=True)
    parser.add_argument('--epochs', help='number of epochs;', type=int, default=100)
    parser.add_argument('--iot', help='number of epochs;', type=bool, default=False)
    parser.add_argument('--batch_size', help='batch size when clients train on data;', type=int, default=10)
    parser.add_argument('--lr', help='learning rate for local optimizers;', type=float, default=-1, required=False)
    return parser.parse_args()

def main():
    print("Parsing args...")
    args = parse_args()
    print("Creating trainer...")
    my_trainer = Trainer(dataset=args.dataset, lr=args.lr, batch_size=args.batch_size, is_iot=args.iot)
    my_trainer.warm_up_model()
    print("Training...")
    my_trainer.train(epochs=args.epochs,
                     batch_size=args.batch_size)
    print("Saving model...")
    my_trainer.save_model()


if __name__ == '__main__':
    main()
