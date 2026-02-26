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

    import numpy as np

    src = np.load(f"./{config.SPLITS_PATH}/train/_en.npy", allow_pickle=True)
    tgt = np.load(f"./{config.SPLITS_PATH}/train/_es.npy", allow_pickle=True)

    pad_id = 0
    lengths = (tgt != pad_id).sum(axis=1)
    print("min tgt len:", lengths.min())
    print("mean tgt len:", lengths.mean())
    print("median tgt len:", np.median(lengths))
    print("p90 tgt len:", np.quantile(lengths, 0.9))
    print("p99 tgt len:", np.quantile(lengths, 0.99))

    print("src dtype:", src.dtype, "shape:", src.shape)
    print("tgt dtype:", tgt.dtype, "shape:", tgt.shape)

    # You want something like int32/int16 and shape (N, seq_len)
    assert src.ndim == 2, "src should be [N, seq_len]"
    assert tgt.ndim == 2, "tgt should be [N, seq_len]"
    assert src.dtype != object, "src is object array (ragged or strings)"
    assert tgt.dtype != object, "tgt is object array (ragged or strings)"
    assert np.issubdtype(src.dtype, np.integer), "src not integer token ids"
    assert np.issubdtype(tgt.dtype, np.integer), "tgt not integer token ids"
    eos_id = tokenizer.sp.eos_id()
    pad_id = tokenizer.sp.pad_id()
    bos_id = tokenizer.sp.bos_id()

    for i in [0, 1, 2, 10, 100]:
        t = tgt[i]

        print("i =", i)
        print("t[:20] =", t[:20])

        # BOS should usually be first token (if you designed it that way)
        print("BOS at start?", t[0] == bos_id)

        # EOS should exist before padding (usually)
        eos_pos = np.where(t == eos_id)[0]
        pad_pos = np.where(t == pad_id)[0]
        print("EOS positions:", eos_pos[:5], "count:", len(eos_pos))
        print("first PAD:", pad_pos[0] if len(pad_pos) else None)

        if len(pad_pos):
            first_pad = pad_pos[0]
            assert np.all(t[first_pad:] == pad_id), (
                "Found non-PAD tokens after padding started"
            )
            if len(eos_pos):
                assert eos_pos[0] < first_pad, (
                    "EOS occurs after padding starts (suspicious)"
                )

        print()

    print(f"eos_id: {eos_id}")
    print(f"pad_id: {pad_id}")
    print(f"bos_id: {bos_id}")

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

    es_ids = tokenizer.encode(
        text="hola como estas hoy en este momento",
        add_bos=False,
        add_eos=False,
        prefix="<es-to-en>",
    )
    en_ids = tokenizer.encode(text="", add_bos=True, add_eos=False)
    es = jnp.array([es_ids], dtype=jnp.int32)  # [1, src_len]
    en = jnp.array([en_ids], dtype=jnp.int32)
    generated_ids = []
    while True:
        logits = model(
            src=es,
            target=en,
            src_mask=None,
            self_mask=None,
            cross_mask=None,
            is_training=False,
        )
        next_token = int(jnp.argmax(logits[0, -1]))
        print("next token:", next_token)
        print("next token:", tokenizer.decode([next_token]))
        if next_token == eos_id:
            break

        generated_ids.append(next_token)

        # 🔹 append new token to decoder input
        en = jnp.concatenate([en, jnp.array([[next_token]], dtype=jnp.int32)], axis=1)

    # decode final sentence
    text = tokenizer.decode(generated_ids)
    print(text)


if __name__ == "__main__":
    test()
