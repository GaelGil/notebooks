import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]


class WSConv2d(nn.Module):
    """Class to implement

    Attributes:
        driver: the browser driver needed for webscraping
        config: a dictionary that holds info for our webscraper

    Methods:
        __init__(self, driver, config)
            Initializes the instance to be ready for scraping

        forward(self, url: str)
            Function to set the url that we will scrape
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        gain: int = 2,
    ):
        """ """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * kernel_size**2)) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x) -> torch.Tensor:
        """ """
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class PixelNorm(nn.Module):
    """Class to implement pixelnorm"""

    def __init__(self):
        """Initializes pixelnorm instance"""
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        """ """
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class ConvBlock(nn.Module):
    """Function to implement a convolution block"""

    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super(ConvBlock, self).__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()
        self.use_pn = use_pixelnorm

    def forward(self, x):
        """Function to pass some features through a convblock

        We have our x (features) we pass that through our first convolution followed by
        a lealy ReLU. Then we pass that through pixel norm if we have it set to true. We
        then go through another convolution again followed by leaky ReLU and pixel norm
        if set to true
        """
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x


class Generator(nn.Module):
    """Class to implement a generator model

    Attributes:
        driver: the browser driver needed for webscraping
        config: a dictionary that holds info for our webscraper

    Methods:
        __init__(self, driver, config)
            Initializes the instance to be ready for scraping

        fade_in(self, url: str)
            Function to set the url that we will scrape

        forward(self, url: str)
            Function to set the url that we will scrape
    """

    def __init__(self, z_dim, in_channels, img_channels=3):
        """Initializes generator model
        We set the initial base model. This is pixelnorm
        followed by a convolution tranpose and leaky ReLU. Then we pass that by

        """
        super(Generator, self).__init__()

        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),  # 1 x 1 to 4 x 4
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

        self.initial_rgb = WSConv2d(
            in_channels, img_channels, kernel_size=1, stride=1, padding=0
        )

        self.prog_blocks, self.rgb_layers = (
            nn.ModuleList([]),
            nn.ModuleList([self.initial_rgb]),
        )

        for i in range(len(factors) - 1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(
                WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0)
            )

    def fade_in(self, alpha, upscaled, generated) -> torch.Tensor:
        """Function to fade in the layers

        We don't append layers to our model. We fade them in by passing them
        ...

        Args:
            alpha:
            upscaled:
            generated:
        """
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x, alpha, steps) -> torch.Tensor:
        """Function to pass a input through our generator model

        We first pass our x (input) through our initial convolutional layers.
        If we are on the first step we simply return our input passed through
        initial rgb. If we are on a next step we will upscalr the output and
        ...

        Args:
            x:
            alpha:
            setps:
        """
        out = self.initial(x)
        if steps == 0:
            return self.initial_rgb(out)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)

        return self.fade_in(alpha, final_upscaled, final_out)


class Discriminator(nn.Module):
    """Class to implement a discriminator model"""

    def __init__(self, in_channels: int, img_channels=3) -> None:
        """Initializes discriminator model"""
        super(Discriminator, self).__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(len(factors) - 1, 0, -1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i - 1])
            self.prog_blocks.append(
                ConvBlock(conv_in_c, conv_out_c, use_pixelnorm=False)
            )
            self.rgb_layers.append(
                WSConv2d(img_channels, conv_in_c, kernel_size=1, stride=1, padding=0)
            )

        self.initial_rgb = WSConv2d(
            img_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.final_block = nn.Sequential(
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, padding=0, stride=1),
        )

    def fade_in(self, out, alpha, downscaled) -> torch.Tensor:
        """ """
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, X):
        """ """
        batch_statistics = (
            torch.std(X, dim=0).mean().repeat(X.shape[0], 1, X.shape[2], X.shape[3])
        )
        return torch.cat([X, batch_statistics], dim=1)

    def forward(self, X, alpha, steps):
        """ """
        cur_step = len(self.prog_blocks) - steps
        out = self.leaky(self.rgb_layers[cur_step](X))
        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(X)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)
        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)


if __name__ == "__main__":
    Z_DIM = 50
    IN_CHANNELS = 256
    gen = Generator(Z_DIM, IN_CHANNELS, img_channels=3)
    dis = Discriminator(IN_CHANNELS, img_channels=3)

    for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        num_steps = int(log2(img_size / 4))
        x = torch.randn((1, Z_DIM, 1, 1))
        z = gen(x, 0.5, steps=num_steps)
        assert z.shape == (1, 3, img_size, img_size)
        out = dis(z, alpha=0.5, steps=num_steps)
        assert out.shape == (1, 1)
        print(f"success at image size: {img_size}")
