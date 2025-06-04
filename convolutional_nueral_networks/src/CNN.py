"""Convolutional Neural Network Implementation in PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    This class implelemnts a discriminator model in a generative adversarial network

    Methods:
        __init__(in_channels, out_channels, kernel_size)

        forward(x)
            Function to feed an input forward through the model

    """
    def __init__(self, in_channels, out_channels, kernel_size):
        """Function to initialize the cnn model
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=kernel_size)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size)
        # got the input for fc1 by passing a sample image
        self.fc1 = nn.Linear(32*28*28, 120)
        self.fc2 = nn.Linear(120, 1)

    def forward(self, x):
        """Function to feed an input forward through the model
        Args:
            x: The input
        """
        # convolution layer -> relu -> maxpool 2x2 (stride 1)
        x = self.pool(F.relu(self.conv1(x)))
        # convolution layer -> relu -> maxpool 2x2 (stride 1)
        x = self.pool(F.relu(self.conv2(x)))
        # convolution layer -> relu -> maxpool 2x2 (stride 1)
        x = self.pool(F.relu(self.conv3(x)))
        # flatten our output from the convolutional layers
        x = torch.flatten(x, 1)
        # first fully connected layer
        x = self.fc1(x)
        # second fully 
        x = self.fc2(x)
        return  x
