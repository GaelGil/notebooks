import jax.numpy as jnp
import optax

# from flax.training import train_state
from flax import nnx
from transformer.Transformer import Transformer
from utils.config import Config


def init_train_state(
    config: Config, src_vocab_size: int, target_vocab_size: int
) -> nnx.TrainState:
    """
    Initialize the train state
    Args:
        config: Config

    Returns:
        train_state.TrainState
    """
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

    _ = model(
        src=dummy_src_input,
        src_mask=dummy_src_mask,
        target=dummy_target_input,
        target_mask=dummy_target_mask,
        is_training=False,
    )

    lr_schedule_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-3,
        warmup_steps=int(1000 * 0.1),
        decay_steps=int(1000 * (1 - 0.1)),
        end_value=1e-5,
    )
    opt_adam_with_schedule = optax.adam(learning_rate=lr_schedule_fn)
    # initliaze the optimizer
    optimizer = nnx.Optimizer(model, opt_adam_with_schedule, wrt=nnx.Param)
    graphdef, params = nnx.split(model)
    state = nnx.TrainState.create(
        graphdef=graphdef,
        params=params,
        opt_state=optimizer.opt_state(params.values),
        tx=optax.GradientTransformation,
    )
    return state
