from absl import logging
import jax
import orbax.checkpoint as ocp

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


    # metrics_handler = ocp.JsonCheckpointHandler()
    #     manager = ocp.CheckpointManager(
    #     root_dir="./checkpoints",
    #     checkpointers={
    #         "state": ocp.PyTreeCheckpointHandler(),
    #         "metrics": metrics_handler
    #     }
    # )

#     manager.save(
#     step=epoch,
#     items={
#         "state": state,         # model/optimizer state
#         "metrics": metrics      # dict of metrics
#     }
# )
    # # checkpoint options
    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=config.MAX_TO_KEEP,
        save_interval_steps=config.SAVE_INTERVAL,
        enable_async_checkpointing=config.ASYNC_CHECKPOINTING,
        best_fn=lambda metrics: metrics['val_accuracy'],
    )   
    # checkpoint manager
    manager = ocp.CheckpointManager(
        directory=config.CHECKPOINT_PATH.resolve(),
        options=checkpoint_options,
    )

    # restore state
    if manager.latest_step(): # check if there is a latest checkpoint
        logging.info("Restoring from latest checkpoint")
        best_step = manager.best_step() # get the best step
        # restore from latest checkpoint
        restored = manager.restore(
            step=best_step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(state),  # provide initial state as template
            ),
        )
        # update state to the restored state
        state = restored.state
    else:
        logging.info("No checkpoint found, training from scratch")
    # train the model
    logging.info("Training the model")
    train(
        state=state,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.EPOCHS,
        manager=manager,
        logger=logging,
    )

    logging.info("Saving the final model")
    params = jax.device_get(state.params)
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(config.FINAL_SAVE_PATH, args=ocp.args.StandardSave(params))


if __name__ == "__main__":
    main()
