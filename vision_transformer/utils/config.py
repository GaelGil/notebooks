from pathlib import Path

from pydantic import BaseModel
from torchvision import transforms


class Config(BaseModel):
    DEVICE: str
    DATA_PATH: str
    BATCH_SIZE: int
    NUM_WORKERS: int
    NUM_PATCHES: int
    MODEL_PATH: str
    NUM_CLASSES: int
    D_MODEL: int
    N: int
    H: int
    D_FF: int
    DROPOUT: float
    IMG_SIZE: int
    PATCH_SIZE: int
    IN_CHANNELS: int
    TRAIN_SPLIT: float
    VAL_SPLIT: float
    EPOCHS: int
    LR: float
    CHECKPOINT_PATH: Path
    FINAL_SAVE_PATH: Path
    MAX_TO_KEEP: int
    SAVE_INTERVAL: int
    ASYNC_CHECKPOINTING: bool
    SPLITS_PATH: Path
    BEST_FN: str


config = Config(
    DEVICE="DEVICE",
    DATA_PATH="./data/images",
    BATCH_SIZE=32,
    NUM_WORKERS=0,
    MODEL_PATH="model.pth",
    NUM_CLASSES=1,
    D_MODEL=512,
    N=3,
    H=8,
    D_FF=2048,
    DROPOUT=0.1,
    PATCH_SIZE=16,
    IMG_SIZE=127,
    NUM_PATCHES=(127 // 16) ** 2,  # 49 patches/sequence of length 49
    IN_CHANNELS=3,
    TRAIN_SPLIT=0.8,
    VAL_SPLIT=0.1,
    EPOCHS=15,
    LR=0.001,
    CHECKPOINT_PATH=Path("./checkpoints"),
    FINAL_SAVE_PATH=Path("./checkpoints/final"),
    MAX_TO_KEEP=5,
    SAVE_INTERVAL=1,
    ASYNC_CHECKPOINTING=False,
    SPLITS_PATH=Path("./data/splits"),
    BEST_FN="val_accuracy",
)


IMG_TRANSFORMATIONS = transforms.Compose(
    [
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),  # convert from NCHW to NHWC
        transforms.Lambda(lambda x: (x - 0.5) / 0.5),  # normalize manually
    ]
)
