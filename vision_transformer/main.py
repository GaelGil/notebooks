import jax
import orbax.checkpoint as ocp
from absl import logging

from utils.config import IMG_TRANSFORMATIONS, config
from utils.ImageDataset import ImageDataset
from utils.init_state import init_state

logging.set_verbosity(logging.INFO)


def main():
    # set the device
    device = jax.devices("gpu")[0]
    logging.info(f"Using device: {device}")

    # initialize the dataset
    logging.info(f"Loading Dataset from: {config.DATA_PATH}")
    dataset = ImageDataset(
        dataset_path=config.DATA_PATH, transformations=IMG_TRANSFORMATIONS
    )
    logging.info(f"Dataset length: {dataset.get_length()}")

    if config.SPLITS_PATH:
        logging.info(f"Loading datset from {config.SPLITS_PATH}")
        train, val, _ = dataset.load_splits(config.SPLITS_PATH)
    else:
        logging.info("Splitting the dataset into train, val and test sets")
        train, val, _ = dataset.split_data(
            train_split=config.TRAIN_SPLIT,
            val_split=config.VAL_SPLIT,
            save_splits_path=config.SPLITS_PATH,
        )

    train_loader = dataset.get_loader(
        dataset=train,
        seed=42,
        batch_size=config.BATCH_SIZE,
        drop_remainder=True,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
    )

    val_loader = dataset.get_loader(
        dataset=val,
        seed=42,
        batch_size=config.BATCH_SIZE,
        drop_remainder=True,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
    )

    # initialize the model
    logging.info("Initializing the model and optimizer")
    # initialize the checkpoint manager options
    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=config.MAX_TO_KEEP,
        save_interval_steps=config.SAVE_INTERVAL,
        enable_async_checkpointing=config.ASYNC_CHECKPOINTING,
        best_fn=lambda metrics: metrics[config.BEST_FN],
        best_mode="min",
    )

    # initialize the checkpoint manager with the options
    manager = ocp.CheckpointManager(
        directory=config.CHECKPOINT_PATH.resolve(),
        options=checkpoint_options,
    )

    logging.info("Initializing the the model state ...")
    model, optimizer, step = init_state(
        config=config,
        manager=manager,
        logger=logging,
    )

    # get the number of batches per epoch

    train(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.EPOCHS,
        manager=manager,
        logger=logging,
        step=step,
    )


if __name__ == "__main__":
    main()
