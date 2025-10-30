import jax
import jax.numpy as jnp

from transformer.model import Transformer
from utils.config import Config
from flax.training import train_state
import optax


def init_train_state(config: Config) -> train_state.TrainState:
    model: Transformer = Transformer(
        num_classes=config.NUM_CLASSES,
        patch_size=config.PATCH_SIZE,
        d_model=config.D_MODEL,
        N=config.N,
        n_heads=config.H,
        dropout=config.DROPOUT,
        img_size=config.IMG_SIZE,
        in_channels=config.IN_CHANNELS,
        d_ff=config.D_FF,
    )

    rng: jax.random.PRNGKey = jax.random.PRNGKey(0)

    dummy_src_input = jnp.zeros(
        (
            config.BATCH_SIZE,
            config.SRC_VOCAB_SIZE,
            config.SEQ_LEN,
            config.D_MODEL,
        ),
        dtype=jnp.int32,
    )
    dummy_src_mask = jnp.zeros((config.BATCH_SIZE, config.SEQ_LEN), dtype=jnp.float32)

    dummy_target_input = jnp.zeros(
        (
            config.BATCH_SIZE,
            config.TARGET_VOCAB_SIZE,
            config.SEQ_LEN,
            config.D_MODEL,
        ),
        dtype=jnp.int32,
    )
    dummy_target_mask = jnp.zeros(
        (config.BATCH_SIZE, config.SEQ_LEN), dtype=jnp.float32
    )

    # Initialize with dummy inputs
    variables = model.init(
        rng,
        src=dummy_src_input,
        src_mask=dummy_src_mask,
        target=dummy_target_input,
        target_mask=dummy_target_mask,
    )

    params = variables["params"]

    # initliaze the optimizer
    optimizer = optax.adam(learning_rate=config.LR)

    # define the train state
    # apply_fn tells flax how to run a forward pass
    # params are the parameters of the model
    # tx is the optimizer used to update the parameters
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )
    return state
