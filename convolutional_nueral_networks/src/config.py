import torch
from torchvision import transforms

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_TRANSFORMATIONS = transforms.Compose(
    [
        transforms.Resize((255, 255)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
DATA_PATH = "./data/PetImages"
IN_CHANNELS = 3
NUM_CLASSES = 1
KERNEL_SIZE = 5
EPOCHS = [30]
LR_RATES = [0.00001, 0.0001, 0.001]
DROPOUT_RATES = [0, 0.25, 0.5]
BATCH_SIZE = 64
NUM_WORKERS = 8
MODEL_PATH = "./models/model_state_dict.pth"
