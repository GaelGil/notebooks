import logging

from utils.config import config
from utils.LangDataset import LangDataset
from utils.TokenizeDataset import TokenizeDataset

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    # set the device
    # device = jax.devices("gpu")[0]
    # logger.info(f"Using device: {device}")

    # load the dataset
    logger.info(f"Loading Dataset from: {config.DATA_PATH}")
    lang_dataset = LangDataset(
        dataset_name=config.DATA_PATH,
        src_lang=config.LANG_SRC,
        target_lang=config.LANG_TARGET,
    )
    logger.info(f"Length of dataset: {lang_dataset.__len__()}")
    lang_dataset.handle_null()
    logger.info(f"Length of dataset after handling null: {lang_dataset.__len__()}")

    # tokenize the dataset in both languages using the entire dataset
    logger.info("Tokenizing the dataset ...")
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

    # get the token ids
    logger.info("Getting the token ids ...")
    src_ids = src_tokenizer.get_token_ids()
    target_ids = target_tokenizer.get_token_ids()

    lang_dataset.set_src_target_ids(src_ids=src_ids, target_ids=target_ids)

    train_loader, val_loader, test_loader = lang_dataset.split_data(
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        batch_size=config.BATCH_SIZE,
    )

    train_laoder, val_loader, test_loader = lang_dataset.get_loaders()

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
    # state: train_state.TrainState = init_train_state(config)

    # # create checkpoint
    # checkpointer = ocp.StandardCheckpointer()

    # # checkpoint options
    # checkpoint_options = ocp.CheckpointManagerOptions(
    #     max_to_keep=config.MAX_TO_KEEP, save_interval_steps=2
    # )
    # # checkpoint manager
    # manager = ocp.CheckpointManager(
    #     directory=config.CHECKPOINT_PATH,
    #     options=checkpoint_options,
    #     handler_registry=checkpointer,
    # )

    # # restore from latest checkpoint if exists
    # if manager.latest_step():
    #     logger.info("Restoring from latest checkpoint")
    #     manager.restore(manager.latest_step())
    # else:
    #     logger.info("No checkpoint found, training from scratch")
    # train(
    #     state=state,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     num_epochs=config.EPOCHS,
    #     manager=manager,
    # )


if __name__ == "__main__":
    main()
