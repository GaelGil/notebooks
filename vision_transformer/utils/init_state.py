import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from absl import logging
from flax import nnx

from utils.config import Config
from vision_transformer.VisionTransformer import VisionTransformer


def init_state(
    config: Config,
    manager: ocp.CheckpointManager,
    logger: logging,
) -> tuple[VisionTransformer, nnx.Optimizer[VisionTransformer], int]:
    """
    Initializes the train state

    Args:
        config: Config

    Returns:
        train_state.TrainState
    """
    # initialize the abstract model
    abs_model = nnx.eval_shape(
        lambda: VisionTransformer(
            num_classes=config.NUM_CLASSES,
            patch_size=config.PATCH_SIZE,
            d_model=config.D_MODEL,
            N=config.N,
            n_heads=config.H,
            d_ff=config.D_FF,
            dropout=config.DROPOUT,
            img_size=config.IMG_SIZE,
            in_channels=config.IN_CHANNELS,
            rngs=nnx.Rngs(0),
        )
    )

    # initialize the abstract optimizer
    abs_opt = nnx.eval_shape(
        lambda: nnx.Optimizer(abs_model, optax.adam(1e-3), wrt=nnx.Param)
    )
    # get the abstract optimizer state
    abs_opt_state = nnx.state(abs_opt)

    # split into graphdef and abstract state
    _graphdef, abs_state, _rng = nnx.split(
        abs_model,
        nnx.Param,  # trainable weights
        nnx.RngState,  # dropout RNGs
    )

    # initalize model and optimizer
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
        rngs=nnx.Rngs(0),
    )

    optimizer = optax.adamw(learning_rate=config.LR)
    optimizer = nnx.Optimizer(model, optimizer, wrt=nnx.Param)
    # create a dummy input to initialize the model
    dummy_input = jnp.zeros(
        (config.BATCH_SIZE, config.IMG_SIZE, config.IMG_SIZE, config.IN_CHANNELS),
        dtype=jnp.float32,
    )

    # restore the state
    step = 0 if manager.best_step() is None else manager.best_step()

    if step is not None and step > 0:
        step = manager.latest_step()
        logger.info(f"Restoring from step {step}")
        # restore the state
        restored = manager.restore(
            step=step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(abs_state),
                optimizer=ocp.args.StandardRestore(abs_opt_state),
            ),
        )
        all_steps = manager.all_steps()
        steps_to_delete = [s for s in all_steps if s > step]  # ie [5,6,7,8,9,10]
        for s in steps_to_delete:
            manager.delete(s)
        # update the model and optimizer with the restored state and optimizer
        nnx.update(optimizer, restored["optimizer"])
        nnx.update(model, restored["state"])
        # return the restored model and optimizer
        return model, optimizer, step

    _ = model(
        x=dummy_input,
        is_training=False,
    )

    return model, optimizer, step
