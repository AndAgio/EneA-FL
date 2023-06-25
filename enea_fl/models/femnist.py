from torch import nn
import torch.nn.functional as t_func
from .config_files import FemnistConfig


class CnnFemnist(nn.Module):
    def __init__(self, ):
        super(CnnFemnist, self).__init__()
        self.config = FemnistConfig()
        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels=self.config.image_shape[0],
                               out_channels=self.config.conv_channels[0],
                               kernel_size=self.config.conv_kernels[0],
                               stride=self.config.conv_strides[0])
        self.conv2 = nn.Conv2d(in_channels=self.config.conv_channels[0],
                               out_channels=self.config.conv_channels[1],
                               kernel_size=self.config.conv_kernels[1],
                               stride=self.config.conv_strides[1])
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=self.config.lin_channels[0],
                             out_features=self.config.lin_channels[1])
        self.fc2 = nn.Linear(in_features=self.config.lin_channels[1],
                             out_features=self.config.n_classes)

    def forward(self, x):
        # x = t_func.max_pool2d(t_func.relu(self.conv1(x)), 2, 2)
        # x = t_func.max_pool2d(t_func.relu(self.conv2(x)), 2, 2)
        x = self.conv1(x)
        print("conv1")
        x = t_func.relu(x)
        x = t_func.max_pool2d(x, 2, 2)
        print("maxpool1")
        x = self.conv2(x)
        x = t_func.relu(x)
        x = t_func.max_pool2d(x, 2, 2)
        x = x.view(-1, self.config.lin_channels[0])
        x = self.fc1(x)
        x = t_func.relu(x)
        # x = t_func.relu(self.fc1(x))
        print("end forward")
        return t_func.softmax(self.fc2(x), dim=1)
