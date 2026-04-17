import orbax.checkpoint as ocp
from jax import numpy as jnp
import numpy as np

from utils.config import config
from utils.init_state import init_state
from pathlib import Path
from utils.Tokenizer import Tokenizer


def test():
    tokenizer = Tokenizer(
        corpus_path=config.SRC_CORPUS_PATH,
        tokenizer_path=config.TOKENIZER_PATH,
        tokenizer_model_path=config.TOKENIZER_MODEL_PATH,
        model_prefix="joint",
        seq_len=config.SEQ_LEN,
    )
    tokenizer.load_tokenizer()

    config.BATCH_SIZE = 12
    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=config.MAX_TO_KEEP,
        save_interval_steps=config.SAVE_INTERVAL,
        enable_async_checkpointing=config.ASYNC_CHECKPOINTING,
        best_fn=lambda metrics: metrics[config.BEST_FN],
        best_mode="min",
    )
    config.CHECKPOINT_PATH = Path(
        "./chckpnts_phase_2_mixed_model_bsize_12_epoch_200/16/"
    )

    manager = ocp.CheckpointManager(
        directory=config.CHECKPOINT_PATH.resolve(),
        options=checkpoint_options,
    )
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size}")
    model, _, step = init_state(
        config=config,
        src_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        manager=manager,
        batches_per_epoch=100,
    )

    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=config.MAX_TO_KEEP,
        save_interval_steps=config.SAVE_INTERVAL,
        enable_async_checkpointing=config.ASYNC_CHECKPOINTING,
        best_fn=lambda metrics: metrics[config.BEST_FN],
        best_mode="min",
    )
    config.CHECKPOINT_PATH = Path(
        "./chckpnts_phase_2_mixed_model_bsize_12_epoch_200/16/"
    )

    manager = ocp.CheckpointManager(
        directory=config.CHECKPOINT_PATH.resolve(),
        options=checkpoint_options,
    )
    vocab_size = tokenizer.get_vocab_size()
    model, _, step = init_state(
        config=config,
        src_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        manager=manager,
        batches_per_epoch=100,
    )
    print(f"Loaded checkpoint at step: {step}")

    sp_test = np.load("./data/splits/test/_sp.npy")
    nah_test = np.load("./data/splits/test/_nah.npy")

    eos_id = tokenizer.sp.eos_id()
    pad_id = tokenizer.sp.pad_id()

    num_samples = min(5, len(sp_test))
    for i in range(num_samples):
        sp = sp_test[i]
        nah = nah_test[i]

        sp_tokens = [int(t) for t in sp if t != pad_id]
        nah_tokens = [int(t) for t in nah if t != pad_id]

        print(f"\n--- Sample {i + 1} ---")
        print(f"Source (sp): {tokenizer.sp.Decode(sp_tokens)}")

        actual_target = tokenizer.sp.Decode(nah_tokens)
        print(f"Actual target (nah): {actual_target}")

        es = jnp.array([sp], dtype=jnp.int32)
        en = jnp.array([[tokenizer.sp.bos_id()]], dtype=jnp.int32)

        generated_ids = []
        for _ in range(128):
            decoder_self_attention_mask = jnp.tril(
                jnp.ones((en.shape[1], en.shape[1]), dtype=jnp.bool_)
            )[None, None, :, :]
            logits = model(
                src=es,
                target=en,
                src_mask=None,
                self_mask=decoder_self_attention_mask,
                cross_mask=None,
                is_training=False,
            )
            next_token = int(jnp.argmax(logits[0, -1]))

            if next_token == eos_id:
                break

            generated_ids.append(next_token)
            en = jnp.concatenate(
                [en, jnp.array([[next_token]], dtype=jnp.int32)], axis=1
            )

        model_output = tokenizer.sp.Decode(generated_ids)
        print(f"Model output:     {model_output}")


if __name__ == "__main__":
    test()
