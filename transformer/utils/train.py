import optax
from flax import nnx

from transformer.model import Transformer
from utils.config import config


def train():
    model: Transformer = Transformer(
        d_model=config.D_MODEL,
        N=config.N,
        n_heads=config.H,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        seq_len=config.SEQ_LEN,
        src_vocab_size=config.SRC_VOCAB_SIZE,
        target_vocab_size=config.TARGET_VOCAB_SIZE,
    )
    optimizer = nnx.Optimizer(optax.adam(learning_rate=config.LR), model=model)
