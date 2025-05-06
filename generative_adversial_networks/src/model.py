import torch 
import torch.nn as nn
import torch.nn.functional as F
from math import log2

factors = [1, 1, 1, 1,  1/2, 1/4, 1/8, 1/16, 1/32]

class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * kernel_size ** 2)) ** 0.5
        self.bias = self.conv.bias

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale ) + self.bias.view(1, self.bias.shape[0], 1, 1)

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8
        
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True)+ self.epsilon)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super().__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()
        self.use_pn = use_pixelnorm

    def forward(self, X):
        X = self.leaky(self.conv1(X))
        X = self.pn(X) if self.use_pn else X
        X = self.leaky(self.conv2(X))
        X = self.pn(X) if self.use_pn else X
        return X


class Generator(nn.Module):
    def __init__(self, z_dim,  in_channels, img_channels=3):
        super().__init__()
        self,initial = nn.Sequential(
            PixelNorm(), 
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0), # 1 x 1 to 4 x 4
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )
        self.initial_rgb = WSConv2d(in_channels, img_channels, kernel_size=1, stride=1)
        self.prog_blocks = nn.ModuleList()
        self.rgb_layers = nn.ModuleList(self.initial_rgb)

        for i in range(len(factors)-1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels* factors[i+1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1))

    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1-alpha) * upscaled)

    def forward(self, X, alpha, steps):
        out = self.initial(X)
        if steps == 0:
            return self.initial_rgb(out)
        
        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode='nearest')
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps-1](upscaled)
        final_out = self.rgb_layers[steps](out)


        return self.fade_in(alpha, final_upscaled, final_out)

class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.prog_block = nn.ModuleList()
        self.rgb_layers = nn.ModuleList()
        self.leaky = nn.LeakyReLU(0.2)

        # for i in range(len(factors)-1, 0, -1):

        return

    def fade_in(self):
        pass
    def minibatch(self, X):
        pass
    def forward(self, X):
        pass