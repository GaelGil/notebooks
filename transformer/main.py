import os

import orbax.checkpoint as ocp
from absl import logging

from utils.CheckpointManager import CheckpointManager
from utils.config import config
from utils.init_train_state import init_train_state
from utils.LangDataset import LangDataset
from utils.Tokenizer import Tokenizer
from utils.train_eval import train

logging.set_verbosity(logging.INFO)


def main():
    # set the device
    # device = jax.devices("gpu")[0]
    # logging.info(f"Using device: {device}")

    tokenizer = Tokenizer(
        joint_corpus_path=config.JOINT_CORPUS_PATH,
        tokenizer_model_path=config.TOKENIZER_MODEL_PATH,
        tokenizer_path=config.TOKENIZER_PATH,
    )

    dataset_one = LangDataset(
        src_file=config.SRC_FILE,
        target_file=config.TARGET_FILE,
        src_lang=config.LANG_SRC_ONE,
        target_lang=config.LANG_TARGET_ONE,
        seq_len=config.SEQ_LEN,
    )

    dataset_two = LangDataset(
        dataset_name=config.DATA_PATH,
        src_lang=config.LANG_SRC_TWO,
        target_lang=config.LANG_TARGET_TWO,
        seq_len=config.SEQ_LEN,
    )

    raw_src_one, raw_target_one = dataset_one.load_data()
    raw_src_two, raw_target_two = dataset_two.load_data()

    if os.path.exists(config.TOKENIZER_MODEL_PATH):
        logging.info("Loading the tokenizer ...")
        tokenizer.load_tokenizer()
    else:
        logging.info("Training the tokenizer ...")
        tokenizer.train_tokenizer(
            src_one=raw_src_one,
            target_one=raw_target_one,
            src_two=raw_src_two,
            target_two=raw_target_two,
        )

    src_one, target_one = dataset_one.prep_data(
        raw_src_one, raw_target_one, tokenizer=tokenizer
    )
    src_two, target_two = dataset_two.prep_data(
        raw_src_two, raw_target_two, tokenizer=tokenizer
    )

    if os.path.exists(config.SPLITS_PATH):
        logging.info("Loading the splits ...")
        src_one_train, src_one_val, target_one_train, target_one_val = (
            dataset_one.load_splits()
        )
        src_two_train, src_two_val, target_two_train, target_two_val = (
            dataset_two.load_splits()
        )
    else:
        logging.info("Splitting the data ...")
        src_one_train, src_one_val, target_one_train, target_one_val = (
            dataset_one.split(
                src=src_one,
                target=target_one,
                train_size=config.TRAIN_SPLIT,
                val_size=config.VAL_SPLIT,
                src_name=config.LANG_SRC_TWO,
                target_name=config.LANG_TARGET_TWO,
                splits_path=config.SPLITS_PATH,
            )
        )
        src_two_train, src_two_val, target_two_train, target_two_val = (
            dataset_two.split(
                src=src_two,
                target=target_two,
                train_size=config.TRAIN_SPLIT,
                val_size=config.VAL_SPLIT,
                src_name=config.LANG_SRC_TWO,
                target_name=config.LANG_TARGET_TWO,
                splits_path=config.SPLITS_PATH,
            )
        )

    train_batch = dataset_one.create_batches(
        src=src_one_train,
        target=target_one_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    val_batch = dataset_one.create_batches(
        src=src_one_val,
        target=target_one_val,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    # initialize the train state
    logging.info("Initializing the train state ...")
    state = init_train_state(config=config)

    # initialize the checkpoint manager
    logging.info("Initializing the checkpoint manager ...")
    checkpoint_manager = CheckpointManager(config=config)

    checkpoint_manager.add_to_register(
        "state", ocp.args.StandardSave, ocp.args.StandardRestore
    )
    checkpoint_manager.add_to_register(
        "metrics", ocp.args.JsonSave, ocp.args.JsonRestore
    )

    # create the checkpoint manager
    logging.info("Creating the checkpoint manager ...")
    checkpoint_manager.create_manager()

    manager = checkpoint_manager.get_manager()

    state, step = checkpoint_manager.restore(state=state, logging=logging)

    # initialize the model
    logging.info("Initializing the model and optimizer")
    state = init_train_state(config)

    # define checkpoint options
    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=config.MAX_TO_KEEP,
        save_interval_steps=config.SAVE_INTERVAL,
        enable_async_checkpointing=config.ASYNC_CHECKPOINTING,
        best_fn=lambda metrics: metrics[config.BEST_FN],
    )

    # Create handler registry
    registry = ocp.handlers.DefaultCheckpointHandlerRegistry()

    # PyTree (model/optimizer state)
    registry.add("state", ocp.args.StandardSave)
    registry.add("state", ocp.args.StandardRestore)

    # JSON (metrics)
    registry.add("metrics", ocp.args.JsonSave)
    registry.add("metrics", ocp.args.JsonRestore)

    # Define the checkpoint manager
    manager = ocp.CheckpointManager(
        directory=config.CHECKPOINT_PATH.resolve(),
        handler_registry=registry,
        options=checkpoint_options,
    )

    # restore previous checkpoint
    if manager.latest_step():  # check if there is a latest checkpoint
        logging.info("Restoring from latest checkpoint")
        # get the best step/checkpoint
        # this was deinfed in the checkpoint options
        best_step = manager.best_step()
        # restore from the best step
        restored = manager.restore(
            step=best_step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(state),
                metrics=ocp.args.JsonRestore(),
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
        train_loader=train_batch,
        val_loader=val_batch,
        epochs=config.EPOCHS,
        manager=manager,
        logger=logging,
    )

    train_batch = dataset_one.create_batches(
        src=src_two_train,
        target=target_two_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    val_batch = dataset_one.create_batches(
        src=src_one_val,
        target=target_two_val,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    logging.info("Training completed, training with new data")
    train(
        state=state,
        train_loader=train_batch,
        val_loader=val_batch,
        epochs=config.EPOCHS,
        manager=manager,
        logger=logging,
    )


if __name__ == "__main__":
    main()
