from transformer.model import Transformer
from utils.config import config


def train():
    print(config)
    model: Transformer = Transformer(
        src_vocab_size=config.SRC_VOCAB_SIZE,
        target_vocab_size=config.TARGET_VOCAB_SIZE,
        seq_len=config.SRC_SEQ_LEN,
        d_model=config.D_MODEL,
        N=config.N,
        n_heads=config.H,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
    )
    return model
