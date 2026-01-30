import jax
import orbax.checkpoint as ocp
from absl import logging

from utils.config import IMG_TRANSFORMATIONS, config
from utils.ImageDataset import ImageDataset
from utils.init_state import init_state
from utils.train_eval import train

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
    logging.info("Splitting the dataset into train, val and test sets")
    # split the dataset
    dataset.split_data(
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        save_splits_path=config.SPLITS_PATH,
    )
    train_loader, val_loader, test_loader = dataset.get_loaders()
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
    batches_per_epoch = train_data.__len__() // config.BATCH_SIZE
    val_batches_per_epoch = val_data.__len__() // config.BATCH_SIZE

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.EPOCHS,
        manager=manager,
        logger=logging,
        step=step,
    )


if __name__ == "__main__":
    main()
