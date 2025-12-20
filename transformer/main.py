import grain
import orbax.checkpoint as ocp
from absl import logging
from grain.samplers import IndexSampler
from grain.transforms import Batch

from utils.config import config
from utils.DataLoader import Source
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

    # initialize the data source
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
    # train_loader: grain.DataLoaderIterator = iter(train_loader)
    # val_loader: grain.DataLoaderIterator = iter(val_loader)

    # initialize the checkpoint manager options
    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=config.MAX_TO_KEEP,
        save_interval_steps=config.SAVE_INTERVAL,
        enable_async_checkpointing=config.ASYNC_CHECKPOINTING,
        best_fn=config.BEST_FN,
    )

    # initialize the checkpoint manager with the options
    manager = ocp.CheckpointManager(
        directory=config.CHECKPOINT_PATH.resolve(),
        options=checkpoint_options,
    )

    logging.info("Initializing the the model state ...")
    model, optimizer = init_state(
        config=config,
        src_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        manager=manager,
        logger=logging,
    )

    batches_per_epoch = train_data.__len__() // config.BATCH_SIZE
    val_batches_per_epoch = val_data.__len__() // config.BATCH_SIZE

    step = 0 if manager.latest_step() is None else manager.latest_step()
    logging.info(f"Training the model from step {step}")
    if step != config.EPOCHS:
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
    train_loader: grain.DataLoaderIterator = iter(train_loader)
    val_loader: grain.DataLoaderIterator = iter(val_loader)

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
