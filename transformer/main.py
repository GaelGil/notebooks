from pathlib import Path

import jax
import orbax.checkpoint as ocp
from absl import logging

from utils.CheckpointManager import CheckpointManager
from utils.config import config
from utils.DataLoader import DataLoader
from utils.init_train_state import init_train_state
from utils.LangDataset import LangDataset
from utils.Tokenizer import Tokenizer
from utils.train_eval import train

logging.set_verbosity(logging.INFO)


def main():
    # set the device
    device = jax.devices("gpu")[0]
    logging.info(f"Using device: {device}")

    tokenizer = Tokenizer(
        joint_corpus_path=config.JOINT_CORPUS_PATH,
        tokenizer_model_path=config.TOKENIZER_MODEL_PATH,
        tokenizer_path=config.TOKENIZER_PATH,
    )

    logging.info("Initializing the datasets ...")
    dataset_one = LangDataset(
        src_file=config.SRC_FILE,
        target_file=config.TARGET_FILE,
        src_lang=config.LANG_SRC_ONE,
        target_lang=config.LANG_TARGET_ONE,
        seq_len=config.SEQ_LEN,
        prefix=config.PREFIXES[0],
    )

    dataset_two = LangDataset(
        dataset_name=config.DATA_PATH,
        src_lang=config.LANG_SRC_TWO,
        target_lang=config.LANG_TARGET_TWO,
        seq_len=config.SEQ_LEN,
        prefix=config.PREFIXES[1],
    )

    logging.info("Loading the data ...")
    raw_src_one, raw_target_one = dataset_one.load_data()
    raw_src_two, raw_target_two = dataset_two.load_data()

    if Path(config.TOKENIZER_MODEL_PATH).exists():
        logging.info("Loading the tokenizer ...")
        tokenizer.load_tokenizer()
    else:
        logging.info("Training the tokenizer ...")
        tokenizer.train_tokenizer(
            src_one=raw_src_one,
            target_one=raw_target_one,
            src_two=raw_src_two,
            target_two=raw_target_two,
            prefixs=config.PREFIXES,
        )

    if Path("./data/splits/test_nah.npy").exists():
        logging.info("Loading the splits ...")
        src_one_train, src_one_val, target_one_train, target_one_val, _, _ = (
            dataset_one.load_splits(
                splits_path=config.SPLITS_PATH,
                src_name=config.LANG_SRC_ONE,
                target_name=config.LANG_TARGET_ONE,
            )
        )
        src_two_train, src_two_val, target_two_train, target_two_val, _, _ = (
            dataset_two.load_splits(
                splits_path=config.SPLITS_PATH,
                src_name=config.LANG_SRC_TWO,
                target_name=config.LANG_TARGET_TWO,
            )
        )
    else:
        logging.info("Prepping the data ...")
        src_one, target_one = dataset_one.prep_data(
            raw_src_one,
            raw_target_one,
            tokenizer=tokenizer,
        )
        src_two, target_two = dataset_two.prep_data(
            raw_src_two,
            raw_target_two,
            tokenizer=tokenizer,
        )
        logging.info("Splitting the data ...")
        src_one_train, src_one_val, target_one_train, target_one_val, _, _ = (
            dataset_one.split(
                src=src_one,
                target=target_one,
                train_size=config.TRAIN_SPLIT,
                val_size=config.VAL_SPLIT,
                src_name=config.LANG_SRC_ONE,
                target_name=config.LANG_TARGET_ONE,
                splits_path=config.SPLITS_PATH,
            )
        )
        src_two_train, src_two_val, target_two_train, target_two_val, _, _ = (
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

    train_loader = DataLoader(
        src=src_one_train,
        target=target_one_train,
        batch_size=config.BATCH_SIZE,
        seq_len=config.SEQ_LEN,
        shuffle=True,
    )

    val_loader = DataLoader(
        src=src_one_val,
        target=target_one_val,
        batch_size=config.BATCH_SIZE,
        seq_len=config.SEQ_LEN,
        shuffle=False,
    )
    # print(train_loader.src.shape)
    # print(val_loader.src.shape)
    print(f"tokenizer vocab size: {tokenizer.vocab_size}")
    print(tokenizer.sp.encode("hello world"))

    # initialize the train state
    logging.info("Initializing the train state ...")
    state = init_train_state(config=config, vocab_size=tokenizer.vocab_size)
    # print("ID 0 =", tokenizer.sp.id_to_piece(0))
    # print("ID 1 =", tokenizer.sp.id_to_piece(1))
    # print("ID 2 =", tokenizer.sp.id_to_piece(2))
    # print("ID 3 =", tokenizer.sp.id_to_piece(3))
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

    logging.info("Training the model")
    if step != config.EPOCHS:
        train(
            state=state,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.EPOCHS,
            manager=manager,
            logger=logging,
            step=step,
        )

    train_loader = DataLoader(
        src=src_two_train,
        target=target_two_train,
        batch_size=config.BATCH_SIZE,
        seq_len=config.SEQ_LEN,
        shuffle=True,
    )
    val_batch = DataLoader(
        src=src_two_val,
        target=target_two_val,
        batch_size=config.BATCH_SIZE,
        seq_len=config.SEQ_LEN,
        shuffle=False,
    )

    logging.info("Training completed, training with new data")
    train(
        state=state,
        train_loader=train_loader,
        val_loader=val_batch,
        epochs=config.EPOCHS,
        manager=manager,
        logger=logging,
    )


if __name__ == "__main__":
    main()
