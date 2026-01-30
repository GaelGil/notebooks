import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from utils.config import Config
from vision_transformer.VisionTransformer import VisionTransformer


def init_state(config: Config) -> train_state.TrainState:
    """
    Initializes the train state

    Args:
        config: Config

    Returns:
        train_state.TrainState
    """
    # initialize the model instance
    model: VisionTransformer = VisionTransformer(
        num_classes=config.NUM_CLASSES,
        patch_size=config.PATCH_SIZE,
        d_model=config.D_MODEL,
        N=config.N,
        n_heads=config.H,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        img_size=config.IMG_SIZE,
        in_channels=config.IN_CHANNELS,
    )
    # initialize the rng
    rng = jax.random.PRNGKey(0)

    # create a dummy input to initialize the model
    dummy_input = jnp.zeros(
        (config.BATCH_SIZE, config.IMG_SIZE, config.IMG_SIZE, config.IN_CHANNELS),
        dtype=jnp.float32,
    )

    # get the parameters of the model
    variables = model.init(
        rng,
        x=dummy_input,
        is_training=True,
    )
    params = variables["params"]

    # initliaze the optimizer
    optimizer = optax.adamw(learning_rate=config.LR)

    # define the train state
    # apply_fn tells jax how to run a forward pass
    # params are the parameters of the model
    # tx is the optimizer used to update the parameters
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    return state
