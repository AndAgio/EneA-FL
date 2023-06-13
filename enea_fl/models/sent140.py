import os
import subprocess
from .config_files import SentConfig
import json
import numpy as np
import torch
import torch.nn as nn


class CnnSent(nn.Module):
    def __init__(self):
        super(CnnSent, self).__init__()

        self.config = SentConfig()

        try:
            with open(self.config.embs_file, 'r') as inf:
                embs = json.load(inf)
        except FileNotFoundError:
            rc = subprocess.call("./enea_fl/models/get_embs.sh", shell=True)
            with open(self.config.embs_file, 'r') as inf:
                embs = json.load(inf)
        vocab = embs['vocab']
        vocab_size = len(vocab)
        word_embeddings = torch.from_numpy(np.array(embs['emba']))
        word_embeddings_size = word_embeddings.shape[1]

        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, word_embeddings_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=word_embeddings_size, out_channels=self.config.num_channels,
                      kernel_size=self.config.kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[0] + 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=word_embeddings_size, out_channels=self.config.num_channels,
                      kernel_size=self.config.kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[1] + 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=word_embeddings_size, out_channels=self.config.num_channels,
                      kernel_size=self.config.kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[2] + 1)
        )

        self.dropout = nn.Dropout(self.config.dropout_keep)

        # Fully-Connected Layer
        self.fc = nn.Linear(self.config.num_channels * len(self.config.kernel_size), self.config.output_size)

        # Softmax non-linearity
        self.softmax = nn.Softmax()

    def forward(self, x):
        # x.shape = (max_sen_len, batch_size)
        embedded_sent = self.embeddings(x).permute(1, 2, 0)
        # embedded_sent.shape = (batch_size=64,embed_size=300,max_sen_len=20)

        conv_out1 = self.conv1(embedded_sent).squeeze(2)  # shape=(64, num_channels, 1) (squeeze 1)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)

        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        final_out = self.fc(final_feature_map)
        return self.softmax(final_out)
