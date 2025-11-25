import jax
import orbax.checkpoint as ocp
from absl import logging

from utils.CheckpointManager import CheckpointManager
from utils.config import IMG_TRANSFORMATIONS, config
from utils.ImageDataset import ImageDataset
from utils.init_train_state import init_train_state
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
    state = init_train_state(config)

    # initialize the checkpoint manager
    checkpoint_manager = CheckpointManager(config)
    #  PyTree (model/optimizer state)
    checkpoint_manager.add_to_register(
        "state", ocp.args.StandardSave, ocp.args.StandardRestore
    )
    # JSON (metrics)
    checkpoint_manager.add_to_register(
        "metrics", ocp.args.JsonSave, ocp.args.JsonRestore
    )
    # create the manager
    checkpoint_manager.create_manager()
    # get the manager
    manager = checkpoint_manager.get_manager()
    # restore the state
    logging.info("Restoring the model")
    state, step = checkpoint_manager.restore(state, logging)
    logging.info("Training the model")
    train(
        state=state,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.EPOCHS,
        manager=manager,
        logger=logging,
        step=step
    )

    logging.info("Saving the final model")
    params = jax.device_get(state.params)
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(
        config.FINAL_SAVE_PATH.resolve(), args=ocp.args.StandardSave(params)
    )


if __name__ == "__main__":
    main()
