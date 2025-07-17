"""This class implements a discriminator in a GAN model in PyTorch."""

import torch.nn as nn
import torch


class Discriminator(nn.Module):
    """
    This class implelemnts a discriminator model in a generative adversarial network

    Attributes:
        discriminator: The discriminator model

    Methods:
        __init__(img_channels, features)

        block(in_channels, out_channels, kernel_size, stride, padding)
            Function to add a sequential block to our model.

        forward(x)
            Function to feed an input forward through the model

    """

    def __init__(
        self,
        img_channels: int,
        features: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        """Initialise discriminator instance

        This function initializes our model with a convolution, followed by leakyReLU, followed
        by 3 convolution blocks (defined in block() function) then a single convolution and a
        sigmoid function

        Args:
            img_channels: The number of channels in the image
            features: The number of features
            kernel_size: size of the kernel
            stride: The stride size
            padding: Padding size

        Returns:
            None
        """
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(
                in_channels=img_channels,
                out_channels=features,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.LeakyReLU(0.2),
            self.block(
                in_channels=features,
                out_channels=features * 2,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            self.block(
                in_channels=features * 2,
                out_channels=features * 4,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            self.block(
                in_channels=features * 4,
                out_channels=features * 8,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.Conv2d(
                in_channels=features * 8,
                out_channels=1,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
            ),
            nn.Sigmoid(),
        )

    def block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> nn.Sequential:
        """Function to add a sequential block to our model.

        This function adds a sequential block to our model. We add a convolutional layer,
        Then we perform batchnorm followed by a leaklyReLU.

        Args:
            in_channels: The number of in channles for this block
            out_channels: The number of out channels for this block
            kernel_size: The kernel size (nxn)
            stirde: The stride to move the kernel by
            padding: If we are adding padding when performing the convolutions

        Returns:
            nn.Sequential
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Function to feed an input forward through the model"""
        return self.discriminator(x)
