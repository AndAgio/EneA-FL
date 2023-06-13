import torch
from torch import nn
import torch.nn.functional as t_func


class CnnFemnist(nn.Module):
    def __init__(self, n_classes=62):
        super(CnnFemnist, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(in_features=5*5*50, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=n_classes)

    def forward(self, x):
        # x = torch.reshape(x, (-1, 1, 28, 28))
        x = t_func.relu(self.conv1(x))
        x = t_func.max_pool2d(x, 2, 2)
        x = t_func.relu(self.conv2(x))
        x = t_func.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*50)
        x = t_func.relu(self.fc1(x))
        x = self.fc2(x)
        return t_func.softmax(x, dim=1)
