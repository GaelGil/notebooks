import logging

import jax
import optax
import orbax.checkpoint as ocp
from flax.training import train_state
from utils.init_train_state import init_train_state
from utils.TokenizeDataset import TokenizeDataset
from utils.config import config
from utils.LangDataset import LangDataset
from utils.train_eval import train

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    # set the device
    device = jax.devices("gpu")[0]
    logger.info(f"Using device: {device}")

    # load the dataset
    logger.info(f"Loading Dataset from: {config.DATA_PATH}")
    dataset_obj = LangDataset()
    dataset = dataset_obj.load_dataset()

    print(f"Dataset length: {dataset_obj.length()}")

    # split the into train, val and test
    train_dataset, val_dataset, test_dataset = dataset_obj.split()
    # print(dataset["train"][0][config.LANG_SRC])
    # tokenize the dataset in both languages using the entire dataset
    tokenizer_src = TokenizeDataset(
        dataset=dataset["train"],
        language=config.LANG_SRC,
        tokenizer_path=config.TOKENIZER_FILE,
    )
    tokenizer_target = TokenizeDataset(
        dataset=dataset["train"],
        language=config.LANG_TARGET,
        tokenizer_path=config.TOKENIZER_FILE,
    )

    # get the tokenizers
    tokenizer_src = tokenizer_src.get_tokenizer()
    tokenizer_target = tokenizer_target.get_tokenizer()

    # logger.info("Splitting the dataset into train, val and test sets")
    # # split the dataset
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
