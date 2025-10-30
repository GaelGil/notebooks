from logging import getLogger

import jax
import optax
import orbax.checkpoint as ocp
from flax.training import train_state
from utils.initialize_model import initialize_model

from utils.config import config
from utils.LangDataset import LangDataset
from utils.train_eval import train

logger = getLogger(__name__)


def main():
    # set the device
    device = jax.devices("gpu")[0]
    logger.info(f"Using device: {device}")

    # initialize the dataset
    logger.info(f"Loading Dataset from: {config.DATA_PATH}")
    dataset = LangDataset(dataset_path=config.DATA_PATH, transformations=None)
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
    state = initialize_model(config)

    # create checkpoint
    checkpointer = ocp.StandardCheckpointer()

    # checkpoint options
    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=config.MAX_TO_KEEP, save_interval_steps=2
    )
    # checkpoint manager
    manager = ocp.CheckpointManager(
        directory=config.CHECKPOINT_PATH,
        options=checkpoint_options,
        handler_registry=checkpointer,
    )

    # restore from latest checkpoint if exists
    if manager.latest_step():
        logger.info("Restoring from latest checkpoint")
        manager.restore(manager.latest_step())
    else:
        logger.info("No checkpoint found, training from scratch")
    train(
        model=model,
        state=state,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.EPOCHS,
        manager=manager,
    )


if __name__ == "__main__":
    main()
