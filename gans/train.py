import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from generative_adversial_network import Discriminator, Generator

# hyper parameters
LR = 0.001
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NOISE_DIM = 100
EPOCHS = 50
FEATURES_DISC = 64
FEATURES_GEN = 64

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load generator and disciminator with some hyperparameters
gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

# set the optimizer for our models
opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
step = 0

# image transformations
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize((255, 255)),
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

gen.train()
disc.train()


def train_model(data, gen, disc, epochs=50):
    """
    Function to train a general adversial network.

    Parameters:
    ----------
    data:
        The data we are using for training.
    gen:
        The generator model.
    disc:
        The discriminator model.
    epochs:
        The number of iterations we are trainig the model for
    
    Returns:
    -------
    None
    """
    for epoch in range(epochs):
        # Target labels not needed! <3 unsupervised
        for batch_idx, (real, _) in enumerate(data):
            real = real.to(device)
            noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
            fake = gen(noise)

            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Print losses occasionally and print to tensorboard
            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(data)}"
                )

    return 0 


train_model(data=data_loader, gen=gen, disc=disc, epochs=EPOCHS)
