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
    # nn.

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

    schedule = transformer_schedule(d_model=config.D_MODEL, warmup=config.WARMUP_STEPS)

    # initliaze the optimizer
    optimizer = nnx.Optimizer(
        tx=optax.adamw(learning_rate=schedule), model=model, wrt=nnx.Param
    )

    # define the train state
    # apply_fn tells flax how to run a forward pass
    # params are the parameters of the model
    # tx is the optimizer used to update the parameters
    state = nnx.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    return state, schedule


def transformer_schedule(d_model=256, warmup=4000):
    scale = d_model**-0.5

    def schedule(step):
        step = jnp.maximum(step, 1)
        return scale * jnp.minimum(step**-0.5, step * warmup**-1.5)

    return schedule
