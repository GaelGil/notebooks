import optax
from flax import nnx

from utils.config import config
from utils.ImageDataset import ImageDataset
from utils.train import train
from vision_transformer.model import VisionTransformer


def main():
    dataset = ImageDataset(dataset_path=config.DATA_PATH, transformations=None)
    train_loader, val_loader, test_loader = dataset.split_data(
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    model: VisionTransformer = VisionTransformer(
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
    optimizer = nnx.Optimizer(model=model, opt=optax.adam(learning_rate=1e-3))

    train(model=model, loader=train_loader, optimizer=optimizer, epochs=config.EPOCHS)


if __name__ == "__main__":
    main()
