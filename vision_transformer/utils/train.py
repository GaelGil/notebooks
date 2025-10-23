from utils.config import config
from utils.utils import build_vision_transformer
from vision_transformer.model import VisionTransformer


def train():
    print(config)
    model: VisionTransformer = build_vision_transformer(
        num_classes=config.NUM_CLASSES,
        patch_size=config.PATCH_SIZE,
        d_model=config.D_MODEL,
        N=config.N,
        h=config.H,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        img_size=config.IMG_SIZE,
        num_patches=config.NUM_PATCHES,
    )
    return model
