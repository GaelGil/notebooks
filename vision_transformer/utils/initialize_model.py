from vision_transformer.model import VisionTransformer
import jax.numpy as jnp
import jax


def initialize_model(config):
    model = VisionTransformer(
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

    rng = jax.random.PRNGKey(0)

    dummy_input_ids = jnp.zeros(
        (config.BATCH_SIZE, config.NUM_PATCHES), dtype=jnp.int32
    )
    dummy_mask = jnp.zeros((config.BATCH_SIZE, config.NUM_PATCHES), dtype=jnp.float32)
    dummy_timestep = jnp.zeros((config.BATCH_SIZE,), dtype=jnp.float32)

    # Initialize with dummy inputs
    variables = model.init(
        rng,
        input_ids=dummy_input_ids,
        mask=dummy_mask,
        timestep=dummy_timestep,
        train=False,
    )

    params = variables["params"]

    return model, params
