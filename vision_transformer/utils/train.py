from utils.config import config
from utils.utils import build_vision_transformer
from vision_transformer.model import VisionTransformer


def train():
    print(config)
    model: VisionTransformer = build_vision_transformer(
        src_vocab_size=config.SRC_VOCAB_SIZE,
        target_vocab_size=config.TARGET_VOCAB_SIZE,
        src_seq_len=config.SRC_SEQ_LEN,
        target_seq_len=config.TARGET_SEQ_LEN,
        patch_size=config.PATCH_SIZE,
        d_model=config.D_MODEL,
        N=config.N,
        h=config.H,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
    )
    return model
