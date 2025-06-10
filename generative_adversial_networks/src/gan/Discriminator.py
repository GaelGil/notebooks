"""This class implements a discriminator in a GAN model in PyTorch.
"""

import torch
import torch.nn as nn

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
    def __init__(self, img_channels, features):
        """This function initializes the discriminator instance
        
        This function initializes our model with a convolutiop, followed by leakyReLU, followed
        by 3 convolution blocks (defined in block()) then a single convolution and a sigmoid function

        Args:
            img_channels:
                The number of channels in the image

            features:
                The number of features after each convolution

        Returns:
            None
        """
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self.block(in_channels=features, out_channels=features*2, kernel_size=4, stride=2, padding=1),
            self.block(in_channels=features*2, out_channels=features*4, kernel_size=4, stride=2, padding=1),
            self.block(in_channels=features*4, out_channels=features*8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_channels=features*8, out_channels=1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )


    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        """Function to add a sequential block to our model.

        This function adds a sequential block to our model. We add a 2d convolutional layer,
        Then we perform batchnorm followed by a leaklyReLU.

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
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )


    def forward(self, x):
        """Function to feed an input forward through the model
        """
        return self.discriminator(x)