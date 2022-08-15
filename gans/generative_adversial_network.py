import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise, img_channels, features):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            self.block(noise, features*16, kernel_size=4, stride=1, padding=0),
            self.block(features*16, features*8, kernel_size=4, stride=2, padding=1),
            self.block(features*8, features*4, kernel_size=4, stride=2, padding=1),
            self.block(features*4, features*2, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(features*2, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.generator(x)

class Discriminator(nn.Module):
    def __init__(self, img_channels, features):
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
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )


    def forward(self, x):
        return self.discriminator(x)



def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

