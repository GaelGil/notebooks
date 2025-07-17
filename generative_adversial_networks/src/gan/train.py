import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Generator import Generator
from Discriminator import Discriminator
from init_weights import initialize_weights
import config as config

# get dataset
dataset = datasets.MNIST(
    root="dataset/", train=True, transform=config.IMG_TRANSFORMATIONS, download=True
)

# load dataset into pytorch
dataloader = DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMORY,
    shuffle=True,
)
# initialize generator and discriminator model
gen = Generator(config.NOISE_DIM, config.CHANNELS_IMG, config.FEATURES_GEN).to(
    config.DEVICE
)
disc = Discriminator(
    config.CHANNELS_IMG,
    config.FEATURES_DISC,
    config.KERNEL_SIZE,
    config.STRIDE,
    config.PADDING,
).to(config.DEVICE)
# initialize weights
initialize_weights(gen)
initialize_weights(disc)
# set the optimizer
opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()
# create some noise
fixed_noise = torch.randn(32, config.NOISE_DIM, 1, 1).to(config.DEVICE)
# sett folders for real and fake samples
writer_real = SummaryWriter("logs/real")
writer_fake = SummaryWriter("logs/fake")
step = 0

# set models to training mode
gen.train()
disc.train()
# trai model
for epoch in range(config.NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(config.DEVICE)
        noise = torch.randn(config.BATCH_SIZE, config.NOISE_DIM, 1, 1).to(config.DEVICE)
        fake = gen(noise)

        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{config.NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
