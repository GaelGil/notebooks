import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from generative_adversial_network import Generator
from PIL import Image
import cv2
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from generative_adversial_network import Discriminator, Generator



LR = 0.001
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3
NOISE_DIM = 100
EPOCHS = 200
FEATURES_DISC = 64
FEATURES_GEN = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)


gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)

# fake_image = gen(fixed_noise)


# npimg = fake_image
# save_image(npimg, 'img1.jpg')


# image transformations
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
    ])

# loading the whole mnist dataset
DATA = torchvision.datasets.MNIST(
    root="./data/",
    train=False,
    download=False,
    transform=TRANSFORM_IMG
)

# create the data loader to accesss our data
data_loader = torch.utils.data.DataLoader(DATA, batch_size=BATCH_SIZE, shuffle=True)


iterations = 0
for epoch in range(EPOCHS):
        # Target labels not needed! <3 unsupervised
    for batch_idx, (real, _) in enumerate(data_loader):
            # real image
        # real = real.to(device)
        # print(_)
            # generated noise
        # noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
            # fake image created by generator
        # fake = gen(noise)
        # print()

        # print(batch_idx)
        if iterations % 10 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}] Batch {batch_idx}/{len(data_loader)}")
        iterations += 1


