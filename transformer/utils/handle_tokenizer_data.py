from utils.config import config
from utils.Tokenizer import Tokenizer
from utils.LangDataset import LangDataset
from pathlib import Path


def handle_tokenizer_data(logging) -> tuple[Tokenizer, dict, dict]:
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

    raw_src_one, raw_target_one = dataset_one.load_data()
    raw_src_two, raw_target_two = dataset_two.load_data()
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
    src_one_train, src_one_val, _, target_one_train, target_one_val, _ = (
        dataset_one.load_splits(
            splits_path=config.SPLITS_PATH,
            src_name=config.LANG_SRC_ONE,
            target_name=config.LANG_TARGET_ONE,
        )
    )
    (
        src_two_train,
        src_two_val,
        _,
        target_two_train,
        target_two_val,
        _,
    ) = dataset_two.load_splits(
        splits_path=config.SPLITS_PATH,
        src_name=config.LANG_SRC_TWO,
        target_name=config.LANG_TARGET_TWO,
    )

    return tokenizer, dataset_one.paths, dataset_two.paths
