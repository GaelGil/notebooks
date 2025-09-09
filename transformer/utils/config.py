from pydantic import BaseModel
import torch
from torchvision import transforms


class Config(BaseModel):
    DEVICE: str
    DATA_PATH: str
    BATCH_SIZE: int
    NUM_WORKERS: int
    MODEL_PATH: str


config = Config(
    DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    DATA_PATH="data",
    BATCH_SIZE=32,
    NUM_WORKERS=8,
    MODEL_PATH="model.pth",
)
