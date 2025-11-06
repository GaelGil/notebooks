import jax
from absl import logging

from utils.config import config
from utils.LangDataset import LangDataset
from utils.TokenizeDataset import TokenizeDataset

logging.set_verbosity(logging.INFO)


def main():
    # set the device
    device = jax.devices("gpu")[0]
    logging.info(f"Using device: {device}")

    # load the dataset
    logging.info(f"Loading Dataset from: {config.DATA_PATH}")
    lang_dataset = LangDataset(
        dataset_name=config.DATA_PATH,
        src_lang=config.LANG_SRC,
        target_lang=config.LANG_TARGET,
    )
    logging.info(f"Length of dataset: {lang_dataset.__len__()}")
    lang_dataset.handle_null()
    logging.info(f"Length of dataset after handling null: {lang_dataset.__len__()}")

    # tokenize the dataset in both languages using the entire dataset
    logging.info("Tokenizing the dataset ...")
    src_tokenizer = TokenizeDataset(
        dataset=lang_dataset.dataset["train"],
        language=config.LANG_SRC,
        tokenizer_path=config.TOKENIZER_FILE,
    )
    target_tokenizer = TokenizeDataset(
        dataset=lang_dataset.dataset["train"],
        language=config.LANG_TARGET,
        tokenizer_path=config.TOKENIZER_FILE,
    )

    # get the token ids for the dataset
    logging.info("Getting the token ids ...")
    src_data = src_tokenizer.get_token_ids()
    target_data = target_tokenizer.get_token_ids()

    # set the token ids for the dataset
    lang_dataset.set_src_target_ids(src_data=src_data, target_data=target_data)

    # split the src and target ids into train, val and test sets
    lang_dataset.split(
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        batch_size=config.BATCH_SIZE,
    )

    # get the loaders
    (
        src_train_loader,
        src_val_loader,
        src_test_loader,
        target_train_loader,
        target_val_loader,
        target_test_loader,
    ) = lang_dataset.get_loaders()

    for batch in src_test_loader:
        print(batch)

    # # logger.info("Splitting the dataset into train, val and test sets")
    # # # split the dataset
    # train_loader, val_loader, test_loader = dataset.split_data(
    #     train_split=config.TRAIN_SPLIT,
    #     val_split=config.VAL_SPLIT,
    #     batch_size=config.BATCH_SIZE,
    #     num_workers=config.NUM_WORKERS,
    # )
    # # initialize the model
    # logger.info("Initializing the model and optimizer")
    # state = init_train_state(config)

    # # define checkpoint options
    # define checkpoint options
    # checkpoint_options = ocp.CheckpointManagerOptions(
    #     max_to_keep=config.MAX_TO_KEEP,
    #     save_interval_steps=config.SAVE_INTERVAL,
    #     enable_async_checkpointing=config.ASYNC_CHECKPOINTING,
    #     best_fn=lambda metrics: metrics[config.BEST_FN],
    # )

    # # Create handler registry
    # registry = ocp.handlers.DefaultCheckpointHandlerRegistry()

    # # PyTree (model/optimizer state)
    # registry.add("state", ocp.args.StandardSave)
    # registry.add("state", ocp.args.StandardRestore)

    # # JSON (metrics)
    # registry.add("metrics", ocp.args.JsonSave)
    # registry.add("metrics",ocp.args.JsonRestore)

    # # Define the checkpoint manager
    # manager = ocp.CheckpointManager(
    #     directory=config.CHECKPOINT_PATH.resolve(),
    #     handler_registry=registry,
    #     options=checkpoint_options,
    # )

    # # restore previous checkpoint
    # if manager.latest_step():  # check if there is a latest checkpoint
    #     logging.info("Restoring from latest checkpoint")
    #     # get the best step/checkpoint
    #     # this was deinfed in the checkpoint options
    #     best_step = manager.best_step()
    #     # restore from the best step
    #     restored = manager.restore(
    #         step=best_step,
    #         args=ocp.args.Composite(
    #             state=ocp.args.StandardRestore(state),
    #             metrics=ocp.args.JsonRestore(),
    #         ),
    #     )
    #     # update state to the restored state
    #     state = restored.state
    # else:
    #     logging.info("No checkpoint found, training from scratch")
    # # train the model
    # logging.info("Training the model")
    # train(
    #     state=state,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     epochs=config.EPOCHS,
    #     manager=manager,
    #     logger=logging,
    # )


if __name__ == "__main__":
    main()
