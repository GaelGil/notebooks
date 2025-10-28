from logging import getLogger

import jax
import optax
import orbax.checkpoint as ocp
from flax.training import train_state
from flax import nnx

from utils.config import config, IMG_TRANSFORMATIONS
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
    dataset = ImageDataset(
        dataset_path=config.DATA_PATH, transformations=IMG_TRANSFORMATIONS
    )
    logger.info(f"Dataset length: {dataset.get_datset_length()}")

    logger.info("Splitting the dataset into train, val and test sets")
    # split the dataset
    dataset.split_data(
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        save_splits_path=config.SPLITS_PATH,
    )
    train_loader, val_loader, test_loader = dataset.get_data_loaders()
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
    optimizer = optax.adam(learning_rate=config.LR)

    # define the train state
    # apply_fn tells jax how to run a forward pass
    # params are the parameters of the model
    # tx is the optimizer used to update the parameters
    state = train_state.TrainState.create(
        apply_fn=model.__call__, params=nnx.state(model, nnx.Param), tx=optimizer
    )

    # checkpoint options
    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=config.MAX_TO_KEEP,
        save_interval_steps=config.SAVE_INTERVAL,
        enable_async_checkpointing=config.ASYNC_CHECKPOINTING,
    )
    # checkpoint manager
    manager = ocp.CheckpointManager(
        directory=config.CHECKPOINT_PATH,
        options=checkpoint_options,
    )

    # restore from latest checkpoint if exists
    if manager.latest_step():
        logger.info("Restoring from latest checkpoint")
        manager.restore(manager.latest_step())
    else:
        logger.info("No checkpoint found, training from scratch")
    # train the model
    logger.info("Training the model")
    train(
        model=model,
        state=state,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.EPOCHS,
        manager=manager,
        logger=logger,
    )

    logger.info("Saving the final model")
    params = jax.device_get(state.params)
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(config.FINAL_SAVE_PATH, args=ocp.args.StandardSave(params))


if __name__ == "__main__":
    main()
