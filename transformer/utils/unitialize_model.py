from transformer.model import Transformer
from utils.config import Config


def initialize_model(config: Config):
    model: Transformer = Transformer(
        num_classes=config.NUM_CLASSES,
        patch_size=config.PATCH_SIZE,
        d_model=config.D_MODEL,
        N=config.N,
        n_heads=config.H,
        dropout=config.DROPOUT,
        img_size=config.IMG_SIZE,
        in_channels=config.IN_CHANNELS,
        d_ff=config.D_FF,
    )
    return model, params
