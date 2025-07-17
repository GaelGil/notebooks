import torch
import torch.nn as nn
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_TRANSFORMATIONS = transforms.Compose(
    [
        transforms.Resize((255, 255)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NOISE_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
KERNEL_SIZE = 4
STRIDE = 2
PADDING = 1
NUM_WORKERS = 8
PIN_MEMORY = True
