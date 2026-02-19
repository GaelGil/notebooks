from pathlib import Path

from pydantic import BaseModel


class Config(BaseModel):
    BATCH_SIZE: int
    EPOCHS: int
    LR: float
    SEQ_LEN: int
    D_MODEL: int
    D_FF: int
    H: int
    N: int
    LANG_SRC_ONE: str
    LANG_TARGET_ONE: str
    LANG_SRC_TWO: str
    LANG_TARGET_TWO: str
    DROPOUT: float
    DATA_PATH: str
    MAX_TO_KEEP: int
    CHECKPOINT_PATH: Path
    TRAIN_SPLIT: float
    VAL_SPLIT: float
    TEST_SPLIT: float
    WORKER_COUNT: int
    BEST_FN: str
    SRC_FILE: Path
    TARGET_FILE: Path
    ASYNC_CHECKPOINTING: bool
    TOKENIZER_PATH: str
    TOKENIZER_MODEL_PATH: str
    SRC_CORPUS_PATH: Path
    TARGET_CORPUS_PATH: Path
    SPLITS_PATH: Path
    SAVE_INTERVAL: int
    PREFIXES: list
    WARMUP_STEPS: int
    SEED: int


config = Config(
    BATCH_SIZE=32,
    EPOCHS=50,
    LR=5e-5,
    SEQ_LEN=128,
    D_MODEL=512,
    D_FF=2048,
    H=8,
    N=6,
    LANG_SRC_ONE="es",
    LANG_TARGET_ONE="en",
    LANG_SRC_TWO="sp",
    LANG_TARGET_TWO="nah",
    DROPOUT=0.1,
    DATA_PATH="somosnlp-hackathon-2022/Axolotl-Spanish-Nahuatl",
    MAX_TO_KEEP=5,
    CHECKPOINT_PATH=Path("./chckpnts_lr"),
    TRAIN_SPLIT=0.8,
    VAL_SPLIT=0.1,
    TEST_SPLIT=0.1,
    WORKER_COUNT=0,
    BEST_FN="eval_loss",
    SRC_FILE=Path("./data/TED2013.en-es.es"),
    TARGET_FILE=Path("./data/TED2013.en-es.en"),
    TOKENIZER_PATH="./tokenizer",
    TOKENIZER_MODEL_PATH="./tokenizer/model",
    SRC_CORPUS_PATH=Path("./tokenizer/corpus/src_corpus"),
    TARGET_CORPUS_PATH=Path("./tokenizer/corpus/target_corpus"),
    ASYNC_CHECKPOINTING=True,
    SPLITS_PATH=Path("./data/splits/"),
    SAVE_INTERVAL=1,
    PREFIXES=["<es_to_en>", "<es_to_nah>"],
    WARMUP_STEPS=50,
    SEED=42,
)
