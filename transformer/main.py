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
    logging.info(f"Using device: {jax.devices('gpu')[0]}")

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

    if config.SPLITS_PATH.exists():
        logging.info("Loading the splits ...")
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

    if not Path(config.TOKENIZER_MODEL_PATH).exists():
        logging.info(
            f"Loading the data from {config.SRC_FILE} and {config.TARGET_FILE} ..."
        )
        logging.info(f"Loading the data from {config.DATA_PATH} ...")
        raw_src_one, raw_target_one = dataset_one.load_data()
        raw_src_two, raw_target_two = dataset_two.load_data()
        logging.info("Training a src and target tokenizer ...")
        tokenizer.train_tokenizer(
            src=raw_src_one,
            target=raw_target_one,
            src_one=raw_src_two,
            target_one=raw_target_two,
            prefixs=config.PREFIXES,
        )

    else:
        logging.info(
            f"Loading the data from {config.SRC_FILE} and {config.TARGET_FILE} ..."
        )
        logging.info(f"Loading the data from {config.DATA_PATH} ...")
        raw_src_one, raw_target_one = dataset_one.load_data()
        raw_src_two, raw_target_two = dataset_two.load_data()
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
        src_one_train, src_one_val, _, target_one_train, target_one_val, _ = (
            dataset_one.split(
                src=src,
                target=target,
                train_size=config.TRAIN_SPLIT,
                val_size=config.VAL_SPLIT,
                src_name=config.LANG_SRC_ONE,
                target_name=config.LANG_TARGET_ONE,
                splits_path=config.SPLITS_PATH,
            )
        )
        (
            src_two_train,
            src_two_val,
            _,
            target_two_train,
            target_two_val,
            _,
        ) = dataset_two.split(
            src=src_one,
            target=target_one,
            train_size=config.TRAIN_SPLIT,
            val_size=config.VAL_SPLIT,
            src_name=config.LANG_SRC_TWO,
            target_name=config.LANG_TARGET_TWO,
            splits_path=config.SPLITS_PATH,
        )

    vocab_size = tokenizer.get_vocab_size()

    train_loader = DataLoader(
        src=src_one_train,
        target=target_one_train,
        batch_size=config.BATCH_SIZE,
        seq_len=config.SEQ_LEN,
        shuffle=True,
        tokenizer=tokenizer,
    )
    val_loader = DataLoader(
        src=src_one_val,
        target=target_one_val,
        batch_size=config.BATCH_SIZE,
        seq_len=config.SEQ_LEN,
        shuffle=False,
    )

    # for batch in train_loader.__iter__(rng=jax.random.PRNGKey(0)):
    #     for i, (src, tgt) in enumerate(zip(batch["src_input"], batch["target_input"])):
    #         input_ids = [int(x) for x in src if int(x) != tokenizer.sp.pad_id()]
    #         target_ids = [int(x) for x in tgt if int(x) != tokenizer.sp.pad_id()]

    #         print(f"SAMPLE {i}")
    #         print("INPUT:", input_ids)
    #         print("INPUT DECODED:", tokenizer.decode(input_ids))
    #         print("TARGET:", target_ids)
    #         print("TARGET DECODED:", tokenizer.decode(target_ids))
    #     break  # remove break if you want to iterate over all batches

    # initialize the train state
    logging.info("Initializing the train state ...")
    state, scheduler = init_train_state(
        config=config,
        src_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
    )

    # initialize the checkpoint manager
    logging.info("Initializing the checkpoint manager ...")
    checkpoint_manager = CheckpointManager(config=config)
    checkpoint_manager.add_to_register(
        "state", ocp.args.StandardSave, ocp.args.StandardRestore
    )
    checkpoint_manager.add_to_register(
        "metrics", ocp.args.JsonSave, ocp.args.JsonRestore
    )

    # assemble the checkpoint manager
    logging.info("Assembling the checkpoint manager ...")
    checkpoint_manager.create_manager()
    manager = checkpoint_manager.get_manager()

    # restore from the latest checkpoint
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
            scheduler=scheduler,
            step=step,
            tokenizer=tokenizer,
            # target_tokenizer=tokenizer,
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
        scheduler=scheduler,
    )


if __name__ == "__main__":
    main()
