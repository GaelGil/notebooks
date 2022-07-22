import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder



class Generator(nn.Module):
    def __init__(self, noise, img_channels, features):
        super(Generator, self).__init__():
        self.net = nn.Sequential(
            nn.ConvTranspose2D(noise, features*16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(features*16),
            nn.ReLU(),


            nn.ConvTranspose2D(features*16, features*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*8),
            nn.ReLU(),


            nn.ConvTranspose2D(features*8, features*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*4),
            nn.ReLU(),


            nn.ConvTranspose2D(features*4, features*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*2),
            nn.ReLU(),

            nn.ConvTranspose2D(noise, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, img_channels, features):
        super(Discriminator, self).__init__():
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLu(0.2),

            nn.Conv2d(in_channels=features, out_channels=features*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLu(0.2),

            nn.Conv2d(in_channels=features*2, out_channels=features*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLu(0.2),

            nn.Conv2d(in_channels=features*4, out_channels=features*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*8),
            nn.LeakyReLu(0.2),

            nn.Conv2d(in_channels=features*8, out_channels=1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

    
    def forward(self, x):
        return self.net(x)