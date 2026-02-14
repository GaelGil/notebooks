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
from jax import numpy as jnp
import numpy as np


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
        logger=logging,
        batches_per_epoch=batches_per_epoch,
    )


    def strip_pad(ids, pad_id=0):
        ids = list(map(int, np.array(ids)))
        if pad_id in ids:
            ids = ids[:ids.index(pad_id)]
        return ids

    batch = next(iter(val_loader))
    enc, dec, labels, labels_mask, enc_pad, dec_self, enc_dec = batch

    logits = model(
        src=enc,
        src_mask=enc_pad,
        target=dec,
        self_mask=dec_self,
        cross_mask=enc_dec,
        is_training=False,
    )

    pred = jnp.argmax(logits, axis=-1)

    for i in range(3):
        src_ids = strip_pad(enc[i], pad_id=0)
        gold_ids = strip_pad(labels[i], pad_id=0)
        pred_ids = strip_pad(pred[i], pad_id=0)

        print("\nSRC :", tokenizer.decode(src_ids))
        print("GOLD:", tokenizer.decode(gold_ids))
        print("PRED:", tokenizer.decode(pred_ids))


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
            step=step,
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
