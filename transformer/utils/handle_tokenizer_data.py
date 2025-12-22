from pathlib import Path

from utils.config import config
from utils.LangDataset import LangDataset
from utils.Tokenizer import Tokenizer
from absl import logging


def handle_tokenizer_data(logging: logging) -> tuple[Tokenizer, dict, dict]:
    """
    Handles the tokenizer and dataset instances

    Args:
        logging: logger instance
    Returns:
        tuple[Tokenizer, dict, dict]: tokenizer, dataset_one_paths, dataset_two_paths"""
    # initialize the src tokenizer instance
    tokenizer = Tokenizer(
        corpus_path=config.SRC_CORPUS_PATH,
        tokenizer_path=config.TOKENIZER_PATH,
        tokenizer_model_path=config.TOKENIZER_MODEL_PATH,
        model_prefix="joint",
        seq_len=config.SEQ_LEN,
    )

    # initialize the dataset instances
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

    # load the data
    raw_src_one, raw_target_one = dataset_one.load_data()
    raw_src_two, raw_target_two = dataset_two.load_data()
    # if the tokenizer model does not exist we train one
    # with the raw data and if not we just load the trained one
    if not Path(config.TOKENIZER_MODEL_PATH).exists():
        logging.info("Training the tokenizer ...")
        tokenizer.train_tokenizer(
            src=raw_src_one,
            target=raw_target_one,
            src_one=raw_src_two,
            target_one=raw_target_two,
            prefixs=config.PREFIXES,
        )
    else:
        logging.info("Loading the tokenizer ...")
        tokenizer.load_tokenizer()

    # if the data splits (train, eval, test) exist we build the paths
    # then return the paths of the splits and the tokenizer
    if Path(config.TOKENIZER_MODEL_PATH).exists() and config.SPLITS_PATH.exists():
        logging.info("Tokenizer and data already exist returning ...")
        dataset_one.buil_save_paths(
            src_name=config.LANG_SRC_ONE,
            target_name=config.LANG_TARGET_ONE,
            splits_path=config.SPLITS_PATH,
        )
        dataset_two.buil_save_paths(
            src_name=config.LANG_SRC_TWO,
            target_name=config.LANG_TARGET_TWO,
            splits_path=config.SPLITS_PATH,
        )
        return tokenizer, dataset_one.paths, dataset_two.paths

    # if the data splits (train, eval, test) do not exist we prep the data
    # then split the data and return the paths of the splits
    if not config.SPLITS_PATH.exists():
        logging.info("Prepping the data ...")
        src, target = tokenizer.prep_data(
            src=raw_src_one,
            target=raw_target_one,
            prefix=config.PREFIXES[0],
        )
        src_one, target_one = tokenizer.prep_data(
            src=raw_src_two,
            target=raw_target_two,
            prefix=config.PREFIXES[1],
        )

        logging.info("Splitting the data ...")
        dataset_one.split(
            src=src,
            target=target,
            train_size=config.TRAIN_SPLIT,
            val_size=config.VAL_SPLIT,
            src_name=config.LANG_SRC_ONE,
            target_name=config.LANG_TARGET_ONE,
            splits_path=config.SPLITS_PATH,
        )
        dataset_two.split(
            src=src_one,
            target=target_one,
            train_size=config.TRAIN_SPLIT,
            val_size=config.VAL_SPLIT,
            src_name=config.LANG_SRC_TWO,
            target_name=config.LANG_TARGET_TWO,
            splits_path=config.SPLITS_PATH,
        )
        dataset_one.load_splits(
            splits_path=config.SPLITS_PATH,
            src_name=config.LANG_SRC_ONE,
            target_name=config.LANG_TARGET_ONE,
        )

        dataset_two.load_splits(
            splits_path=config.SPLITS_PATH,
            src_name=config.LANG_SRC_TWO,
            target_name=config.LANG_TARGET_TWO,
        )

    return tokenizer, dataset_one.paths, dataset_two.paths
