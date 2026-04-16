import orbax.checkpoint as ocp
from jax import numpy as jnp

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

    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=config.MAX_TO_KEEP,
        save_interval_steps=config.SAVE_INTERVAL,
        enable_async_checkpointing=config.ASYNC_CHECKPOINTING,
        best_fn=lambda metrics: metrics[config.BEST_FN],
        best_mode="min",
    )
    config.CHECKPOINT_PATH = Path("./chckpnts_phase2_mixed_model/")

    # initialize the checkpoint manager with the options
    manager = ocp.CheckpointManager(
        directory=config.CHECKPOINT_PATH.resolve(),
        options=checkpoint_options,
    )
    # get the vocab size
    vocab_size = tokenizer.get_vocab_size()
    model, _, step = init_state(
        config=config,
        src_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        manager=manager,
        batches_per_epoch=100,
    )
    print(f"STEP: {step}")
    eos_id = tokenizer.sp.eos_id()

    es_ids = tokenizer.encode(
        # text="hola, ¿cual es la capital de Mexico?",
        text="muchas flores son blancas",
        add_bos=False,
        add_eos=False,
        prefix="<es_to_en>",
    )
    en_ids = tokenizer.encode(text="", add_bos=True, add_eos=False)
    es = jnp.array([es_ids], dtype=jnp.int32)  # [1, src_len]
    en = jnp.array([en_ids], dtype=jnp.int32)
    generated_ids = []
    while True:
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
        print("next token:", next_token)
        print("next token:", tokenizer.decode([next_token]))
        if next_token == eos_id:
            break

        generated_ids.append(next_token)

        en = jnp.concatenate([en, jnp.array([[next_token]], dtype=jnp.int32)], axis=1)

    # decode final sentence
    text = tokenizer.decode(generated_ids)
    print(text)


if __name__ == "__main__":
    test()
