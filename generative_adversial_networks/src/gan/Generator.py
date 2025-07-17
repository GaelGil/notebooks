"""This class implements a generator in a GAN model in PyTorch."""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    This class implelemnts a Generator model in a generative adversarial network

    Attributes:
        generator a sequential module that is the generator model

    Methods:
        __init__(noise, img_channels, features)

        block(in_channels, out_channels, kernel_size, stride, padding)
            Function to add a sequential block to our model.

        forward(x)
            Function to feed an input forward through the model

    """

    def __init__(self, noise: torch.Tensor, img_channels: int, features: int) -> None:
        """This function initializes the generator instance"""
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            self.block(noise, features * 16, kernel_size=4, stride=1, padding=0),
            self.block(features * 16, features * 8, kernel_size=4, stride=2, padding=1),
            self.block(features * 8, features * 4, kernel_size=4, stride=2, padding=1),
            self.block(features * 4, features * 2, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(
                features * 2, img_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
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

        This function adds a sequential block to our model. We add a 2d convolutional layer,
        Then we perform batchnorn followed by a leaklyReLU.

        Args:
            in_channels: The number of in channles for this block
            out_channels: The number of out channels for this block
            kernel_size: The kernel size (nxn)
            stirde: The stride to move the kernel by
            padding: If we are adding padding when performing the convolutions

        Returns:
            None
        """
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Function to feed an input forward through the model"""
        return self.generator(x)
