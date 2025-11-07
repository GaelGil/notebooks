"""Convolutional Neural Network Implementation in PyTorch"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VALE(nn.Module):
    """
    This class implelemnts a discriminator model in a generative adversarial network


    Attributes:
        conv1:
            The first convolutional layer
        pool:
            The pooling layer. (gets reused for each layer)
        conv2:
            The second convolutional layer
        conv3:
            The third convolutional layer
        fc1:
            The first fully connected layer
        fc2:
            The second fully connected layer
        fc3:
            The final fully connected layer

    Methods:
        __init__(in_channels, out_channels, kernel_size)

        forward(x)
            Function to feed an input forward through the model
    """

    def __init__(
        self, in_channels, num_classes, kernel_size, dropout_rate, padding=0, stride=1
    ) -> None:
        """Function to initialize the model

        Args:
            in_channels: The number of input chanels from our inputs
            classes: the number of classes/outputs
            kernel_size: the size of the kernels/matrices

        Returns:
            None
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(28 * 28 * 32, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, x) -> torch.Tensor:
        """Function to feed an input forward through the model"""
        return x
