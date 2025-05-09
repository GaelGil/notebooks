import cv2
import torch
from math import log2

START_TRAIN_AT_IMG_SIZE = 128
DATASET = ''
CHECKPOINT_GEN = 'generator.pth'
CHECKPOINT_CRITIC = 'critic.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 1e-3
BATCH_SIZES = [32, 32, 32, 16, 16, 16, 16, 8, 4]
CHANNELS_IMG = 3
Z_DIM = 256
IN_CHANNELS = 256
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1)
NUM_WORKERS = 4