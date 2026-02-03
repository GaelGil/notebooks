import jax.numpy as jnp
import optax

from flax import nnx
from transformer.Transformer import Transformer
from utils.config import Config
import orbax.checkpoint as ocp
from absl import logging


def init_state(
    config: Config,
    src_vocab_size: int,
    target_vocab_size: int,
    manager: ocp.CheckpointManager,
    logger: logging,
) -> tuple[Transformer, nnx.Optimizer]:
    """
    Initialize the state from a checkpoint or create a new one
    Args:
        config: Config

    Returns:
       tuple[Transformer, nnx.Optimizer]
    """

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

    # create abstract model
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
    # we dont need to save the optimizer for our purpose but its nice to have it
    # if we wanted to save the optimizer we could do it like this. also schedule
    # causes issues with checkpointing/restoring
    # # create abstract optimizer
    # abs_opt = nnx.eval_shape(
    #     lambda: nnx.Optimizer(abs_model, optax.adam(1e-3), wrt=nnx.Param)
    # )
    # # get the optimizer state
    # abs_opt_state = nnx.state(abs_opt)

    # split the abstract model into graphdef, state and rng
    _graphdef, abs_state, _rng = nnx.split(
        abs_model,
        nnx.Param,  # trainable weights
        nnx.RngState,  # dropout RNGs
    )

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

    # warmup_steps = int((config.EPOCHS * 0.1) * 100)
    # lr_schedule_fn = optax.join_schedules(
    #     schedules=[
    #         optax.linear_schedule(
    #             init_value=0.0,
    #             end_value=3e-4,
    #             transition_steps=warmup_steps,
    #         ),
    #         optax.constant_schedule(3e-4),
    #     ],
    #     boundaries=[warmup_steps],
    # )

    # create optimizer
    opt_adam_with_schedule = optax.adam(
        learning_rate=config.LR,
        b1=0.9,
        b2=0.98,
        eps=1e-9,
    )
    optimizer = nnx.Optimizer(model, opt_adam_with_schedule, wrt=nnx.Param)

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
                # optimizer=ocp.args.StandardRestore(abs_opt_state),
            ),
        )
        # all_steps = manager.all_steps()
        # steps_to_delete = [s for s in all_steps if s > step]  # [5,6,7,8,9,10]
        # for s in steps_to_delete:
        #     manager.delete(s)
        # update the model and optimizer with the restored state and optimizer
        # nnx.update(optimizer, restored["optimizer"])
        nnx.update(model, restored["state"])
        # return the restored model and optimizer
        return model, optimizer, step

    # run the model with dummy inputs
    _ = model(
        src=dummy_src_input,
        src_mask=dummy_src_mask,
        target=dummy_target_input,
        self_mask=dummy_target_mask,
        cross_mask=dummy_src_mask,
        is_training=False,
        rngs=rngs,
    )

    return model, optimizer, step
