from pydantic import BaseModel


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


config = Config(
    DEVICE="DEVICE",
    DATA_PATH="./data",
    BATCH_SIZE=32,
    NUM_WORKERS=8,
    MODEL_PATH="model.pth",
    NUM_CLASSES=100,
    D_MODEL=512,
    N=6,
    H=8,
    D_FF=2048,
    DROPOUT=0.1,
    PATCH_SIZE=16,
    IMG_SIZE=32,
    NUM_PATCHES=(32 // 16) ** 2,
    IN_CHANNELS=3,
    TRAIN_SPLIT=0.8,
    VAL_SPLIT=0.1,
    EPOCHS=10,
    LR=0.001,
)
