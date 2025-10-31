from pydantic import BaseModel


class Config(BaseModel):
    BATCH_SIZE: int
    EPOCHS: int
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
    PRELOAD: None
    TOKENIZER_FILE: str
    EXPERIMENT_NAME: str
    DROPOUT: float
    DEVICE: str
    DATA_PATH: str
    MAX_TO_KEEP: int
    CHECKPOINT_PATH: str
    TRAIN_SPLIT: float
    VAL_SPLIT: float
    TEST_SPLIT: float


config = Config(
    BATCH_SIZE=32,
    EPOCHS=10,
    LR=10e-4,
    SEQ_LEN=350,
    D_MODEL=512,
    SRC_VOCAB_SIZE=8000,
    TARGET_VOCAB_SIZE=8000,
    D_FF=2048,
    H=8,
    N=6,
    LANG_SRC="sp",
    LANG_TARGET="nah",
    PRELOAD=None,
    TOKENIZER_FILE="tokenizer_{}.json",
    EXPERIMENT_NAME="runs/model",
    DROPOUT=0.1,
    DEVICE="CUDA",
    DATA_PATH="somosnlp-hackathon-2022/Axolotl-Spanish-Nahuatl",
    MAX_TO_KEEP=5,
    CHECKPOINT_PATH="./checkpoints",
    TRAIN_SPLIT=0.9,
    VAL_SPLIT=0.05,
    TEST_SPLIT=0.05,
)
