from pathlib import Path

import jax
import orbax.checkpoint as ocp
from absl import logging

from utils.config import config

from utils.init_state import init_state
from utils.LangDataset import LangDataset
from utils.Tokenizer import Tokenizer
from utils.train_eval import train
import grain
from grain.samplers import IndexSampler
from grain.transforms import Batch

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

    source = Source(num_samples=len(src_one_train))
    train_sampler = IndexSampler(
        num_records=len(source),
        shard_options=grain.sharding.NoSharding,
        shuffle=True,
        num_epochs=1,
        seed=42,
    )

    eval_sampler = IndexSampler(
        num_records=len(source),
        shard_options=grain.sharding.NoSharding,
        shuffle=False,
        num_epochs=1,
        seed=42,
    )

    train_loader = grain.DataLoader(
        data_source=source,
        sampler=train_sampler,
        operations=[Batch(batch_size=config.BATCH_SIZE, drop_remainder=True)],
        worker_count=config.WORKER_COUNT,
    )

    val_loader = grain.DataLoader(
        data_source=source,
        sampler=eval_sampler,
        operations=[Batch(batch_size=config.BATCH_SIZE, drop_remainder=False)],
        worker_count=config.WORKER_COUNT,
    )

    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=config.MAX_TO_KEEP,
        save_interval_steps=config.SAVE_INTERVAL,
        enable_async_checkpointing=config.ASYNC_CHECKPOINTING,
        best_fn=config.BEST_FN,
    )

    manager = ocp.CheckpointManager(
        directory=config.CHECKPOINT_PATH.resolve(),
        options=checkpoint_options,
    )
    # initialize the train state
    logging.info("Initializing the train state ...")
    model, optimizer = init_state(
        config=config,
        src_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        manager=manager,
    )
    step = manager.latest_step()
    # initialize the checkpoint manager
    logging.info("Initializing the checkpoint manager ...")

    logging.info("Training the model")
    if step != config.EPOCHS:
        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.EPOCHS,
            manager=manager,
            logger=logging,
            step=step,
            tokenizer=tokenizer,
        )

    train_loader = grain.DataLoader(
        data_source=source,
        sampler=train_sampler,
        operations=[Batch(batch_size=config.BATCH_SIZE, drop_remainder=True)],
        worker_count=config.WORKER_COUNT,
    )

    val_loader = grain.DataLoader(
        data_source=source,
        sampler=eval_sampler,
        operations=[Batch(batch_size=config.BATCH_SIZE, drop_remainder=False)],
        worker_count=config.WORKER_COUNT,
    )

    logging.info("Training completed, training with new data")
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.EPOCHS,
        manager=manager,
        logger=logging,
        step=step,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    main()
