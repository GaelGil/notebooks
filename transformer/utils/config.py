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
    LANG_SRC_ONE: str
    LANG_TARGET_ONE: str
    LANG_SRC_TWO: str
    LANG_TARGET_TWO: str
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
    NUM_WORKERS: int
    BEST_FN: str
    SRC_FILE: str
    TARGET_FILE: str
    ASYNC_CHECKPOINTING: bool
    TOKENIZER_PATH: str
    TOKENIZER_MODEL_PATH: str
    JOINT_CORPUS_PATH: str
    SPLITS_PATH: str


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
    LANG_SRC_ONE="en",
    LANG_TARGET_ONE="es",
    LANG_SRC_TWO="sp",
    LANG_TARGET_TWO="nah",
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
    NUM_WORKERS=0,
    BEST_FN="val_accuracy",
    SRC_FILE="./data/TED2013.en-es.es",
    TARGET_FILE="./data/TED2013.en-es.en",
    TOKENIZER_PATH="./tokenizer",
    TOKENIZER_MODEL_PATH="./tokenizer/joint.model",
    ASYNC_CHECKPOINTING=True,
    JOINT_CORPUS_PATH="./tokenizer/joint_corpus.txt",
    SPLITS_PATH="./data/splits",
)
