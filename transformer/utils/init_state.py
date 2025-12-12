import jax.numpy as jnp
import optax

from flax import nnx
from transformer.Transformer import Transformer
from utils.config import Config
import orbax.checkpoint as ocp


def init_state(
    config: Config,
    src_vocab_size: int,
    target_vocab_size: int,
    manager: ocp.CheckpointManager,
) -> tuple[Transformer, nnx.Optimizer]:
    """
    Initialize the train state
    Args:
        config: Config

    Returns:
       tuple[Transformer, nnx.Optimizer]
    """
    # craete model abstraction
    abs_transformer: Transformer = Transformer(
        d_model=config.D_MODEL,
        N=config.N,
        n_heads=config.H,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        seq_len=config.SEQ_LEN,
        src_vocab_size=src_vocab_size,
        target_vocab_size=target_vocab_size,
        rngs=nnx.Rngs(0),
    )

    # create dummy inputs
    dummy_src_input = jnp.zeros(
        (config.BATCH_SIZE, config.SEQ_LEN),
        dtype=jnp.int32,
    )

    dummy_src_mask = jnp.ones(
        (config.BATCH_SIZE, 1, 1, config.SEQ_LEN), dtype=jnp.float32
    )

    dummy_target_input = jnp.zeros(
        (config.BATCH_SIZE, config.SEQ_LEN - 1),
        dtype=jnp.int32,
    )

    dummy_target_mask = jnp.ones(
        (config.BATCH_SIZE, 1, 1, config.SEQ_LEN - 1), dtype=jnp.float32
    )
    # get the absctract model state
    # graphdef and state

    abs_model = nnx.eval_shape(
        lambda: Transformer(
            d_model=config.D_MODEL,
            N=config.N,
            n_heads=config.H,
            d_ff=config.D_FF,
            dropout=config.DROPOUT,
            seq_len=config.SEQ_LEN,
            src_vocab_size=src_vocab_size,
            target_vocab_size=target_vocab_size,
            rngs=nnx.Rngs(0),
        )
    )
    abs_opt = nnx.eval_shape(
        lambda: nnx.Optimizer(abs_model, optax.adam(1e-3), wrt=nnx.Param)
    )

    _graphdef, abs_state, _rng = nnx.split(
        abs_model,
        nnx.Param,  # trainable weights
        nnx.RngState,  # dropout RNGs
    )
    abs_opt_state = nnx.state(abs_opt)

    # create model
    rngs = nnx.Rngs(0)
    model: Transformer = Transformer(
        d_model=config.D_MODEL,
        N=config.N,
        n_heads=config.H,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        seq_len=config.SEQ_LEN,
        src_vocab_size=src_vocab_size,
        target_vocab_size=target_vocab_size,
        rngs=rngs,
    )
    # define the learning rate schedule
    lr_schedule_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-3,
        warmup_steps=int(1000 * 0.1),
        decay_steps=int(1000 * (1 - 0.1)),
        end_value=1e-5,
    )

    # create optimizer
    opt_adam_with_schedule = optax.adam(learning_rate=lr_schedule_fn)
    optimizer = nnx.Optimizer(model, opt_adam_with_schedule, wrt=nnx.Param)

    # restore the state
    step = manager.latest_step()
    if step is not None:
        restored = (
            manager.restore(
                step=step,
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(abs_state),
                    optimizer=ocp.args.StandardRestore(abs_opt_state),
                ),
            ),
        )

        nnx.update(optimizer, restored["optimizer"])
        nnx.update(model, restored["state"])
        return model, optimizer

    # run the model
    _ = model(
        src=dummy_src_input,
        src_mask=dummy_src_mask,
        target=dummy_target_input,
        target_mask=dummy_target_mask,
        is_training=False,
    )

    return model, optimizer
