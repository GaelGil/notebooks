import grain
import orbax.checkpoint as ocp
from absl import logging
from grain.samplers import IndexSampler
from grain.transforms import Batch
from pathlib import Path

from utils.config import config
from utils.Source import Source
from utils.MixedDataset import MixedDataset
from utils.handle_tokenizer_data import handle_tokenizer_data
from utils.init_state import init_state
from utils.train_eval import train

logging.set_verbosity(logging.INFO)


def main():
    # get the tokenizer and dataset paths
    tokenizer, dataset_one_paths, dataset_two_paths = handle_tokenizer_data(
        logging=logging
    )

    # get the vocab size
    vocab_size = tokenizer.get_vocab_size()

    # initialize the data source with the paths to the source and target data
    train_data = Source(
        src_path=dataset_one_paths["train_src"],
        target_path=dataset_one_paths["train_target"],
        pad_id=tokenizer.sp.pad_id(),
    )
    val_data = Source(
        src_path=dataset_one_paths["val_src"],
        target_path=dataset_one_paths["val_target"],
        pad_id=tokenizer.sp.pad_id(),
    )

    # initialize the index sampler
    train_sampler = IndexSampler(
        num_records=train_data.__len__(),
        shard_options=grain.sharding.NoSharding(),
        shuffle=True,
        num_epochs=config.EPOCHS,
        seed=42,
    )
    eval_sampler = IndexSampler(
        num_records=val_data.__len__(),
        shard_options=grain.sharding.NoSharding(),
        shuffle=False,
        num_epochs=1,
        seed=42,
    )

    # initialize the dataloader
    train_loader = grain.DataLoader(
        data_source=train_data,
        sampler=train_sampler,
        operations=[Batch(batch_size=config.BATCH_SIZE, drop_remainder=True)],
        worker_count=config.WORKER_COUNT,
    )
    val_loader = grain.DataLoader(
        data_source=val_data,
        sampler=eval_sampler,
        operations=[Batch(batch_size=config.BATCH_SIZE, drop_remainder=False)],
        worker_count=config.WORKER_COUNT,
    )

    # initialize the checkpoint manager options
    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=config.MAX_TO_KEEP,
        save_interval_steps=config.SAVE_INTERVAL,
        enable_async_checkpointing=config.ASYNC_CHECKPOINTING,
        best_fn=lambda metrics: metrics[config.BEST_FN],
        best_mode="min",
    )

    # initialize the checkpoint manager with the options
    manager = ocp.CheckpointManager(
        directory=config.CHECKPOINT_PATH.resolve(),
        options=checkpoint_options,
    )

    # get the number of batches per epoch
    batches_per_epoch = train_data.__len__() // config.BATCH_SIZE
    val_batches_per_epoch = val_data.__len__() // config.BATCH_SIZE

    logging.info("Initializing the the model state ...")
    model, optimizer, step = init_state(
        config=config,
        src_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        manager=manager,
        batches_per_epoch=batches_per_epoch,
    )

    logging.info(f"Training the model from step {step}")
    if step < config.EPOCHS:
        train(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.EPOCHS,
            manager=manager,
            logger=logging,
            batches_per_epoch=batches_per_epoch,
            val_batches_per_epoch=val_batches_per_epoch,
            step=step,
            seed=config.SEED,
            dropout_schedule=config.DROPOUT_SCHEDULE,
        )

    logging.info("Setting up Phase 2")
    es_nah_data = Source(
        src_path=dataset_two_paths["train_src"],  # Spanish
        target_path=dataset_two_paths["train_target"],  # Nahuatl
        pad_id=tokenizer.sp.pad_id(),
    )

    # Validation data: Use Nahuatl only for evaluation
    val_data_phase2 = Source(
        src_path=dataset_two_paths["val_src"],
        target_path=dataset_two_paths["val_target"],
        pad_id=tokenizer.sp.pad_id(),
    )
    # Update config for Phase 2
    config.DROPOUT_SCHEDULE = {0: 0, 15: 0.05, 20: 0.1, 30: 0.15, 45: 0.25, 60: 0.30}
    config.CHECKPOINT_PATH = Path("./chckpnts_phase2_mixed_model/")
    config.BATCH_SIZE = 16

    # Create mixed dataset for training
    train_data_phase2 = MixedDataset(
        en_data=train_data,
        nah_data=es_nah_data,
    )

    # New samplers
    train_sampler = IndexSampler(
        num_records=len(train_data_phase2),
        shard_options=grain.sharding.NoSharding(),
        shuffle=True,
        num_epochs=config.EPOCHS,
        seed=42,
    )
    eval_sampler = IndexSampler(
        num_records=len(val_data_phase2),
        shard_options=grain.sharding.NoSharding(),
        shuffle=False,
        num_epochs=1,
        seed=42,
    )

    # Initialize New dataloaders
    train_loader = grain.DataLoader(
        data_source=train_data_phase2,
        sampler=train_sampler,
        operations=[Batch(batch_size=config.BATCH_SIZE, drop_remainder=True)],
        worker_count=config.WORKER_COUNT,
    )
    val_loader = grain.DataLoader(
        data_source=val_data_phase2,
        sampler=eval_sampler,
        operations=[Batch(batch_size=config.BATCH_SIZE, drop_remainder=False)],
        worker_count=config.WORKER_COUNT,
    )

    batches_per_epoch = len(train_data_phase2) // config.BATCH_SIZE
    val_batches_per_epoch = len(val_data_phase2) // config.BATCH_SIZE

    # Initialize new checkpoint manager
    manager_phase2 = ocp.CheckpointManager(
        directory=config.CHECKPOINT_PATH.resolve(),
        options=checkpoint_options,
    )
    # Initialize new optimizer
    _, optimizer, step = init_state(
        config=config,
        src_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        manager=manager_phase2,
        batches_per_epoch=batches_per_epoch,
    )

    logging.info("Starting Phase 2")
    train(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.EPOCHS,
        manager=manager_phase2,
        logger=logging,
        batches_per_epoch=batches_per_epoch,
        val_batches_per_epoch=val_batches_per_epoch,
        step=step,
        seed=config.SEED,
        dropout_schedule=config.DROPOUT_SCHEDULE,
    )

    # TODO: test seq_len=512
    # TODO: only train on nah and en at the same time using mixed dataset only


if __name__ == "__main__":
    main()
