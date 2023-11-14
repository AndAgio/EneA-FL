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
from enea_fl.models import CnnSent, CnnFemnist, SentConfig, Nbaiot
from enea_fl.utils import get_logger, get_free_gpu
from enea_fl.nbaiot_utils import read_nbaiot_data, nbaio_train_single_epoch, nbaio_test

# SENT140
# sleeping_time_selection = {
#     0: 30,
#     1: 60,
#     2: 30,
#     3: 0,
#     4: 0,
#     5: 30,
# }

# FEMNIST
# sleeping_time_selection = {
#     0: 0,
#     1: 30,
#     2: 30,
#     3: 60,
#     4: 0,
#     5: 0,
# }

# NBAIOT
sleeping_time_selection = {
    0: 0,
    1: 30,
    2: 0,
    3: 30,
    4: 60,
    5: 0,
}

def get_model(dataset, nbaiot_size):
    if dataset == 'femnist':
        return CnnFemnist()
    elif dataset == 'sent140':
        return CnnSent()
    elif dataset == 'nbaiot':
        return Nbaiot(nbaiot_size["train_loader"].dataset.tensors[0].shape[-1])
    else:
        raise ValueError('Dataset "{}" is not available!'.format(dataset))
    
def get_optimizer(dataset, model, lr):
    if dataset == 'nbaiot':
        return optim.SGD(params=model.parameters(),lr=0.1)
    else:
        return optim.SGD(params=model.parameters(), lr=lr)

class Trainer:
    def __init__(self, dataset='femnist', lr=0.01, batch_size=10, is_iot=True, test_size=0.33, cpu=False):
        assert dataset in ['femnist', 'sent140', 'nbaiot']
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.indexization = None
        self.final_data = None
        self.logger = get_logger(node_type='non_fl', node_id='0', log_folder=os.path.join('logs', dataset))
        self.logger.print_it('Istantiating a Trainer object for {} dataset!'.format(dataset))
        self.dataset = dataset
        self.batch_size = batch_size
        self.is_iot = is_iot
        self.test_size = test_size
        self.cpu = cpu
        self.logger.print_it('Reading data asd...')
        try:
            self.processing_device = self.choose_device()
        except Exception as e:
            print(e)
            print("Error during reading data!")
            return
        if self.dataset == 'nbaiot':
            self.logger.print_it('NBAIOT dataset!')
            try:
                self.train_data, self.test_data, self.final_data = read_nbaiot_data("data/nbaiot", self.processing_device, self.batch_size, self.test_size, self.is_iot)
            except Exception as e:
                print(e)
                print("Error during reading data!")
                return
        else:
            self.read_data_from_dir()
        self.logger.print_it('Cleaning previous logger!')
        self.clean_previous_logger()
        self.logger.print_it('Istantiating a model!')
        self.model = get_model(dataset, self.final_data)
        self.logger.print_it('Model: {}'.format(self.model))
        self.logger.print_it('Using a {} for training and inference!'.format(self.processing_device))
        self.model = self.model.to(self.processing_device)
        self.lr = lr
        self._optimizer = get_optimizer(self.dataset, self.model, self.lr)
        self.criterion = nn.CrossEntropyLoss()
        if self.dataset == 'sent140':
            self.indexization = self.gather_indexization()

        self.epoch_timestamps = []
        self.sample_per_epochs = []

    def choose_device(self):
        chosen_device =  torch.device('cpu')
        if not self.cpu:
            chosen_device = torch.device('cuda:{}'.format(get_free_gpu()) if torch.cuda.is_available()
                                else 'cpu')
        self.logger.print_it('Chosen device: {}'.format(chosen_device))
        return chosen_device

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
        self.logger.print_it('[test_size] {}'.format(self.test_size))
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
                                                            test_size=self.test_size,
                                                            random_state=42)
        self.logger.print_it('Gathering xs and ys on single dictionary...')
        train_data = {'x': x_train, 'y': y_train}
        test_data = {'x': x_test, 'y': y_test}
        return train_data, test_data

    def train(self, epochs=100, batch_size=10, simulate_selection=False):
        self.logger.print_it(' Start training '.center(60, '-'))
        self.epoch_timestamps.append(time.time())

        for epoch in range(epochs):
            self.logger.print_it(' | Epoch: {}/{} | LR = {:.5f} | BATCH = {} | '.format(epoch + 1,
                                                                                        epochs,
                                                                                        self.lr,
                                                                                        batch_size).center(60, '-'))
            # Simulate server model training on selected clients' data
            try:
                if self.dataset == 'nbaiot':
                    nbaio_train_single_epoch(self)
                else:
                    final_loss, _, _, _ = self.train_single_epoch(batch_size=batch_size)
            except Exception as e:
                self.logger.print_it('Error during training: {}'.format(e))
                continue
            self.logger.print_it(''.center(60, '-'))
            self.epoch_timestamps.append(time.time())
            if simulate_selection:
                # sleep a random time to simulate client selection for 10 seconds
                self.logger.print_it(' Simulating client selection... sleeping for {} seconds '.format(sleeping_time_selection[epoch]).center(60, '-'))
                self.logger.print_it(''.center(60, '-'))
                time.sleep(sleeping_time_selection[epoch])
        self.logger.print_it(' Training finished! '.center(60, '-'))


    def train_single_epoch(self, batch_size=10):
        start = time.time()
        self.model.train()
        running_loss = 0.
        counter = 0.
        predictions = []
        labels_list = []
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
        self.sample_per_epochs.append(counter)
        energy = 0.  # TODO: find how to compute energy here
        comp = 0.  # TODO: find how to compute flops of model
        return final_loss, energy, stop - start, comp

    def test_model(self, batch_size=10):
        if self.dataset == 'nbaiot':
            return nbaio_test(self.model, self.test_data, self.criterion)

        predictions = []
        labels_list = []
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.
            counter = 0
            for batch_input, batch_label in batch_data(self.test_data, batch_size):
                batch_input, batch_label = self.preprocess_input_output(batch_input, batch_label)
                output = self.model(batch_input)
                # parse batch_label to long (raspberry pi)
                batch_label = batch_label.long()
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
            _, indd, _ = get_word_emb_arr(self.model.embs)
        except Exception as e:
            print(e)
            # _ = subprocess.call("./enea_fl/models/get_embs.sh", shell=True)
            _, indd, _ = get_word_emb_arr(self.model.embs)
        return indd

    def print_message(self, index_batch, batch_size, metrics, mode='train'):
        message = '|'
        bar_length = 10
        if self.dataset == 'nbaiot':
            total_samples = len(self.train_data) if mode == 'train' else len(self.test_data)
            total_batches = math.ceil(total_samples)
        else:
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
            message += (str(index_batch) + "/" + str(total_batches))
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

        if self.dataset == 'nbaiot':
            nbaio_train_single_epoch(self, warm_up=True)
        else:
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
    parser.add_argument('--dataset', help='name of dataset;', type=str, choices=['sent140', 'femnist', 'nbaiot'], required=True)
    parser.add_argument('--epochs', help='number of epochs;', type=int, default=100)
    parser.add_argument('--test_size', help='test size;', type=float, default=0.33)
    parser.add_argument('--iot', help='is this iot;', type=str, default="True")
    parser.add_argument('--cpu', help='number of epochs;', type=str, default="False")
    parser.add_argument('--simulate_selection', help='simulate client selection;', type=str, default="False")
    parser.add_argument('--batch_size', help='batch size when clients train on data;', type=int, default=10)
    parser.add_argument('--lr', help='learning rate for local optimizers;', type=float, default=-1, required=False)
    return parser.parse_args()

def main():
    print("Parsing args...")
    args = parse_args()
    print("args", args)
    print("Creating trainer...")
    my_trainer = Trainer(dataset=args.dataset, lr=args.lr, batch_size=args.batch_size, is_iot=(args.iot == "True"), test_size=args.test_size, cpu=(args.cpu == "True"))
   
    print("Warming up model...")
    try:
        my_trainer.warm_up_model()
    except Exception as e:
        print(e)
        print("Error during warm up!")
        return

    print("Training...")
    my_trainer.train(epochs=args.epochs,
                    batch_size=args.batch_size,
                    simulate_selection=(args.simulate_selection=="True"))
    print("----------------- my_trainer.epoch_timestamps -----------------")
    for i, e in enumerate(my_trainer.epoch_timestamps):
        print(i, ")", e)
    print("----------------- my_trainer.sample_per_epochs -----------------")
    for i, e in enumerate(my_trainer.sample_per_epochs):
        print(i, ")", e)
    print("-------------------------")
    print("Done!")
    print("-------------------------")

    metrics = nbaio_test(my_trainer)
    print("metrics", metrics)


if __name__ == '__main__':
    main()