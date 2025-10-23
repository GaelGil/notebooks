from pydantic import BaseModel


class Config(BaseModel):
    DEVICE: str
    DATA_PATH: str
    BATCH_SIZE: int
    NUM_WORKERS: int
    MODEL_PATH: str
    NUM_CLASSES: int
    SRC_VOCAB_SIZE: int
    TARGET_VOCAB_SIZE: int
    SRC_SEQ_LEN: int
    TARGET_SEQ_LEN: int
    D_MODEL: int
    N: int
    H: int
    D_FF: int
    DROPOUT: float
    PATCHES: int
    PATCH_SIZE: int


config = Config(
    DEVICE="DEVICE",
    DATA_PATH="data",
    BATCH_SIZE=32,
    NUM_WORKERS=8,
    MODEL_PATH="model.pth",
    NUM_CLASSES=100,
    SRC_VOCAB_SIZE=100,
    TARGET_VOCAB_SIZE=100,
    SRC_SEQ_LEN=100,
    TARGET_SEQ_LEN=100,
    D_MODEL=512,
    N=6,
    H=8,
    D_FF=2048,
    DROPOUT=0.1,
    PATCHES=16,
    PATCH_SIZE=16,
)
