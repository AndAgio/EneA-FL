import subprocess
from .config_files import SentConfig
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as t_func


class CnnSent(nn.Module):
    def __init__(self, embs=None):
        super(CnnSent, self).__init__()
        self.config = SentConfig()
        if embs is None:
            try:
                print("Loading GloVe embeddings [{}]...".format(self.config.embs_file))
                with open(self.config.embs_file, 'r') as inf:
                    embs = json.load(inf)
            except FileNotFoundError:
                _ = subprocess.call("./enea_fl/models/get_embs.sh", shell=True)
                with open(self.config.embs_file, 'r') as inf:
                    embs = json.load(inf)
        self.embs = embs
        print("Loaded GloVe embeddings.")
        vocab = self.embs['vocab']
        vocab_size = len(vocab)
        print("Vocab size: {}".format(vocab_size))
        word_embeddings = torch.from_numpy(np.array(self.embs['emba'])).type(torch.FloatTensor)
        word_embeddings_size = word_embeddings.shape[1]
        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, word_embeddings_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
        # Convolution Layers
        self.conv1 = nn.Conv1d(in_channels=word_embeddings_size, out_channels=self.config.cnn_num_channels,
                               kernel_size=self.config.cnn_kernel_size[0])
        self.kernel_size1 = self.config.max_sen_len - self.config.cnn_kernel_size[0] + 1
        self.conv2 = nn.Conv1d(in_channels=word_embeddings_size, out_channels=self.config.cnn_num_channels,
                               kernel_size=self.config.cnn_kernel_size[1])
        self.kernel_size2 = self.config.max_sen_len - self.config.cnn_kernel_size[1] + 1
        self.conv3 = nn.Conv1d(in_channels=word_embeddings_size, out_channels=self.config.cnn_num_channels,
                               kernel_size=self.config.cnn_kernel_size[2])
        self.kernel_size3 = self.config.max_sen_len - self.config.cnn_kernel_size[2] + 1
        self.dropout = nn.Dropout(self.config.dropout)
        # Fully connected layer
        self.fc = nn.Linear(self.config.cnn_num_channels * len(self.config.cnn_kernel_size), self.config.output_size)

    def forward(self, x):
        # Embed input to GloVe embeddings
        embedded_sent = self.embeddings(x)
        embedded_sent = embedded_sent.permute(1, 2, 0)
        # First convolution
        out1 = t_func.max_pool1d(t_func.relu(self.conv1(embedded_sent)), kernel_size=self.kernel_size1).squeeze(2)
        # Second convolution
        out2 = t_func.max_pool1d(t_func.relu(self.conv2(embedded_sent)), kernel_size=self.kernel_size3).squeeze(2)
        # Third convolution
        out3 = t_func.max_pool1d(t_func.relu(self.conv3(embedded_sent)), kernel_size=self.kernel_size3).squeeze(2)
        # Aggregate convolutions
        all_out = torch.cat((out1, out2, out3), 1)
        # Fully connected and output
        final_feature_map = self.dropout(all_out)
        return t_func.softmax(self.fc(final_feature_map), dim=1)


class LstmSent(nn.Module):
    def __init__(self, embs=None):
        super(LstmSent, self).__init__()
        self.config = SentConfig()
        if embs is None:
            try:
                print("Loading GloVe embeddings [{}]...".format(self.config.embs_file))
                with open(self.config.embs_file, 'r') as inf:
                    embs = json.load(inf)
            except FileNotFoundError:
                _ = subprocess.call("./enea_fl/models/get_embs.sh", shell=True)
                with open(self.config.embs_file, 'r') as inf:
                    embs = json.load(inf)
        self.embs = embs
        print("Loaded GloVe embeddings.")
        vocab = self.embs['vocab']
        vocab_size = len(vocab)
        print("Vocab size: {}".format(vocab_size))
        word_embeddings = torch.from_numpy(np.array(self.embs['emba'])).type(torch.FloatTensor)
        word_embeddings_size = word_embeddings.shape[1]
        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, word_embeddings_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)

        self.encoder = nn.LSTM(
            input_size=word_embeddings_size,
            hidden_size=self.config.lstm_hidden,
            num_layers=self.config.lstm_layers,
            bidirectional=self.config.lstm_bidirectional,
            dropout=self.config.dropout,
            batch_first=True
        )

        if self.config.lstm_bidirectional:
            self.config.lstm_hidden *= 2
        self.fc = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.lstm_hidden, self.config.output_size),
        )

    def forward(self, input_seq: torch.Tensor):
        # print('input_seq.shape: {}'.format(input_seq.shape))
        embedded_sent = self.embeddings(input_seq)
        # print('embedded_sent.shape: {}'.format(embedded_sent.shape))
        embedded_sent = embedded_sent.permute(1, 0, 2)
        # print('embedded_sent.shape: {}'.format(embedded_sent.shape))
        lstm_out, _ = self.encoder(embedded_sent)
        final_hidden_state = lstm_out[:, -1]
        output = self.fc(final_hidden_state)
        return t_func.softmax(output, dim=1)


SentModel = CnnSent
# SentModel = LstmSent
