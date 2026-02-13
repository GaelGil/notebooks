from utils.config import config
from utils.DataLoader import Source
from utils.handle_tokenizer_data import handle_tokenizer_data
from utils.init_state import init_state
from absl import logging
import orbax.checkpoint as ocp
from jax import numpy as jnp

def test():
    tokenizer, dataset_one_paths, dataset_two_paths = handle_tokenizer_data(
        logging=logging
    )


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

    # get the vocab size
    vocab_size = tokenizer.get_vocab_size()
    model, _, _ = init_state(
        config=config,
        src_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        manager=manager,
        logger=logging,
        batches_per_epoch=100,
    )
    eos_id = tokenizer.sp.eos_id()

    es_ids = tokenizer.encode(text="como estas", add_bos=True, add_eos=True)
    en_ids = tokenizer.encode(text="", add_bos=True, add_eos=False)
    es = jnp.array([es_ids], dtype=jnp.int32)  # [1, src_len]
    en = jnp.array([en_ids], dtype=jnp.int32)
    generated_ids = []
    while True:
        logits = model(src=es, target=en, src_mask=None, self_mask=None, cross_mask=None, is_training=False)
        next_token = int(jnp.argmax(logits[0, -1]))

        if next_token == eos_id:
            break

        generated_ids.append(next_token)

        # 🔹 append new token to decoder input
        en = jnp.concatenate(
            [en, jnp.array([[next_token]], dtype=jnp.int32)],
            axis=1
        )

    # decode final sentence
    text = tokenizer.decode(generated_ids)
    print(text)

if __name__ == "__main__":
    test()