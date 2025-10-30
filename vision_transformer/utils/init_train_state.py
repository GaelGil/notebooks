from vision_transformer.model import VisionTransformer
import jax.numpy as jnp
import jax
import optax
from utils.config import Config
from flax.training import train_state



def init_train_state(config: Config) -> train_state.TrainState:
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
        training=False
    )

    rng = jax.random.PRNGKey(0)

    dummy_input = jnp.zeros(
    (config.BATCH_SIZE, config.IMG_SIZE, config.IMG_SIZE, config.IN_CHANNELS),
    dtype=jnp.float32,
    )

    # Initialize with dummy inputs
    variables = model.init(
        rng,
        x=dummy_input,
    )

    params = variables["params"]

    # initliaze the optimizer
    optimizer = optax.adam(learning_rate=config.LR)

    # define the train state
    # apply_fn tells jax how to run a forward pass
    # params are the parameters of the model
    # tx is the optimizer used to update the parameters
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    return state
