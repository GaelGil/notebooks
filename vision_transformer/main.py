from logging import getLogger

import jax
import orbax.checkpoint as ocp

from utils.config import IMG_TRANSFORMATIONS, config
from utils.ImageDataset import ImageDataset
from utils.init_train_state import init_train_state
from utils.train_eval import train

logger = getLogger(__name__)


def main():
    # set the device
    device = jax.devices("gpu")[0]
    print(f"Using device: {device}")

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
    state = init_train_state(config)

    # checkpoint options
    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=config.MAX_TO_KEEP,
        save_interval_steps=config.SAVE_INTERVAL,
        enable_async_checkpointing=config.ASYNC_CHECKPOINTING,
    )
    # checkpoint manager
    manager = ocp.CheckpointManager(
        directory=config.CHECKPOINT_PATH.resolve(),
        options=checkpoint_options,
    )

    # restore from latest checkpoint if exists
    if manager.latest_step():
        logger.info("Restoring from latest checkpoint")
        state = manager.restore(
            manager.latest_step(),
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(state),
            ),
        )
    else:
        logger.info("No checkpoint found, training from scratch")
    # train the model
    logger.info("Training the model")
    train(
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
