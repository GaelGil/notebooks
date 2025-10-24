from logging import getLogger

import jax
import optax
from flax import nnx
from flax.training import train_state

from utils.config import config
from utils.ImageDataset import ImageDataset
from utils.train_eval import train
from vision_transformer.model import VisionTransformer

logger = getLogger(__name__)


def main():
    # set the device
    device = jax.devices("gpu")[0]
    logger.info(f"Using device: {device}")

    # initialize the dataset
    logger.info(f"Loading Dataset from: {config.DATA_PATH}")
    dataset = ImageDataset(dataset_path=config.DATA_PATH, transformations=None)
    logger.info(f"Dataset length: {dataset.get_datset_length()}")

    logger.info("Splitting the dataset into train, val and test sets")
    # split the dataset
    train_loader, val_loader, test_loader = dataset.split_data(
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    # initialize the model
    logger.info("Initializing the model and optimizer")
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
    # initliaze the optimizer
    optimizer = nnx.Optimizer(model=model, opt=optax.adam(learning_rate=1e-3))
    # train the model

    logger.info("Training the model")
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=model.params, tx=optimizer
    )
    train(
        model=model,
        state=state,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=config.EPOCHS,
    )


if __name__ == "__main__":
    main()
