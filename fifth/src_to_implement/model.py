import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torchvision as tv
import operator


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Conv2d(in_channels, out_channels)
        self.bn1 = nn.BatchNorm(out_channels)
        self.bn2 = nn.BatchNorm(out_channels)
        self.relu = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(3, 2)

    def ResBlock(self):
        pass

        self.fc1 = nn.Linear(512, 2)
