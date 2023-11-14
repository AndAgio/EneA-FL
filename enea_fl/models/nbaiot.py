import torch.nn.functional as F
import torch.nn as nn

class NbaiotModel(nn.Module):
    def __init__(self, input_shape=29, nb_classes=11):
        super(NbaiotModel, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, nb_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)
    