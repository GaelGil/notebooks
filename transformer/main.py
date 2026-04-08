import grain
import orbax.checkpoint as ocp
from absl import logging
from grain.samplers import IndexSampler
from grain.transforms import Batch
<<<<<<< HEAD

from utils.config import config
from utils.DataLoader import Source
=======
from pathlib import Path

from utils.config import config
from utils.Source import Source
from utils.MixedDataset import MixedDataset
>>>>>>> 91619b06fb7749c0bbd68a49b8340a41d0707956
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
<<<<<<< HEAD
        logger=logging,
=======
>>>>>>> 91619b06fb7749c0bbd68a49b8340a41d0707956
        batches_per_epoch=batches_per_epoch,
    )

    logging.info(f"Training the model from step {step}")
<<<<<<< HEAD
    if step != config.EPOCHS:
=======
    if step < config.EPOCHS:
>>>>>>> 91619b06fb7749c0bbd68a49b8340a41d0707956
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
<<<<<<< HEAD
        )

    # update the dataset paths
    train_data.src = dataset_two_paths["train_src"]
    train_data.target = dataset_two_paths["train_target"]

    val_data.src = dataset_two_paths["val_src"]
    val_data.target = dataset_two_paths["val_target"]

    # update the index sampler
    train_sampler._num_records = train_data.__len__()
    eval_sampler._num_records = val_data.__len__()

    # initialize the dataloaders
    train_loader = grain.DataLoader(
        data_source=train_data,
=======
            dropout_schedule=config.DROPOUT_SCHEDULE,
        )

    # ========== PHASE 2: Mixed Training (80% Nahuatl, 20% English) ==========
    logging.info("Setting up Phase 2: Mixed training (80% Nahuatl, 20% English)")

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
    config.EPOCHS = 50
    config.DROPOUT_SCHEDULE = {0: 0.1, 10: 0.15, 30: 0.2, 45: 0.25, 60: 0.3}
    config.CHECKPOINT_PATH = Path("./chckpnts_phase2_mixed/")
    # config.LR = 2e-4

    # Create mixed dataset for training (80% Nahuatl, 20% English)
    train_data_phase2 = MixedDataset(
        en_data=train_data,
        nah_data=es_nah_data,
    )

    # Update samplers
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

    # Initialize dataloaders
    train_loader = grain.DataLoader(
        data_source=train_data_phase2,
>>>>>>> 91619b06fb7749c0bbd68a49b8340a41d0707956
        sampler=train_sampler,
        operations=[Batch(batch_size=config.BATCH_SIZE, drop_remainder=True)],
        worker_count=config.WORKER_COUNT,
    )
    val_loader = grain.DataLoader(
<<<<<<< HEAD
        data_source=val_data,
=======
        data_source=val_data_phase2,
>>>>>>> 91619b06fb7749c0bbd68a49b8340a41d0707956
        sampler=eval_sampler,
        operations=[Batch(batch_size=config.BATCH_SIZE, drop_remainder=False)],
        worker_count=config.WORKER_COUNT,
    )

<<<<<<< HEAD
    batches_per_epoch = train_data.__len__() // config.BATCH_SIZE
    val_batches_per_epoch = val_data.__len__() // config.BATCH_SIZE

    logging.info("Training completed, training with new data")
=======
    batches_per_epoch = len(train_data_phase2) // config.BATCH_SIZE
    val_batches_per_epoch = len(val_data_phase2) // config.BATCH_SIZE

    manager_phase2 = ocp.CheckpointManager(
        directory=config.CHECKPOINT_PATH.resolve(),
        options=checkpoint_options,
    )
    model, optimizer, step = init_state(
        config=config,
        src_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        manager=manager_phase2,
        batches_per_epoch=batches_per_epoch,
    )

    logging.info("Starting Phase 2: Mixed training (80% Nahuatl + 20% English)")
>>>>>>> 91619b06fb7749c0bbd68a49b8340a41d0707956
    train(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.EPOCHS,
<<<<<<< HEAD
        manager=manager,
=======
        manager=manager_phase2,
>>>>>>> 91619b06fb7749c0bbd68a49b8340a41d0707956
        logger=logging,
        batches_per_epoch=batches_per_epoch,
        val_batches_per_epoch=val_batches_per_epoch,
        step=step,
        seed=config.SEED,
<<<<<<< HEAD
    )

=======
        dropout_schedule=config.DROPOUT_SCHEDULE,
    )

    # TODO: test seq_len=512
    # TODO: only train on nah and en at the same time using mixed dataset only

>>>>>>> 91619b06fb7749c0bbd68a49b8340a41d0707956

if __name__ == "__main__":
    main()
