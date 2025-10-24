from pydantic import BaseModel


class Config(BaseModel):
    BATCH_SIZE: int
    NUM_EPOCHS: int
    LR: float
    SEQ_LEN: int
    D_MODEL: int
    SRC_VOCAB_SIZE: int
    TARGET_VOCAB_SIZE: int
    D_FF: int
    H: int
    N: int
    LANG_SRC: str
    LANG_TARGET: str
    MODEL_FOLDER: str
    MODEL_BASENAME: str
    PRELOAD: None
    TOKENIZER_FILE: str
    EXPERIMENT_NAME: str
    DROPOUT: float
    DEVICE: str
    DATA_PATH: str


config = Config(
    BATCH_SIZE=32,
    NUM_EPOCHS=10,
    LR=10e-4,
    SEQ_LEN=350,
    D_MODEL=512,
    SRC_VOCAB_SIZE=8000,
    TARGET_VOCAB_SIZE=8000,
    D_FF=2048,
    H=8,
    N=6,
    LANG_SRC="span",
    LANG_TARGET="nau",
    MODEL_FOLDER="./models",
    MODEL_BASENAME="tmodel_",
    PRELOAD=None,
    TOKENIZER_FILE="tokenizer.json",
    EXPERIMENT_NAME="runs/model",
    DROPOUT=0.1,
    DEVICE="CUDA",
    DATA_PATH="./data",
)
