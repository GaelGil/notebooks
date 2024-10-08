import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from generative_adversial_network import Discriminator, Generator, initialize_weights

# hyper parameters
LR = 2e-4 
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NOISE_DIM = 100
EPOCHS = 100
FEATURES_DISC = 64
FEATURES_GEN = 64
DATASET_NAME = 'MNIST'


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load generator and disciminator with some hyperparameters
gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

# set the optimizers for our models
opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))
# set a loss
criterion = nn.BCELoss()

# fixed noise to generate some images
fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
step = 0

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

gen.train()
disc.train()


def train_model(data, gen, disc, dataset_name, epochs=50):
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
            # real image
            real = real.to(device)
            # generated noise
            noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
            # fake image created by generator
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

            # Print what iteration we are on along with batch index
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(data)}')
                with torch.no_grad():
                    if epoch % 5 == 0:
                        # generate some images
                        fake_images = gen(fixed_noise)
                        # save them to a folder
                        save_image(fake_images, f'./generated/{dataset_name}/train/image{epoch}.jpg')

    return 0 


train_model(data=data_loader, gen=gen, disc=disc, dataset_name=DATASET_NAME, epochs=EPOCHS)
torch.save(gen.state_dict(), f'./models/{DATASET_NAME}')
