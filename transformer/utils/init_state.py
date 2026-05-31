import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx

from transformer.Transformer import Transformer
from utils.config import Config


def init_state(
    config: Config,
    src_vocab_size: int,
    target_vocab_size: int,
    manager: ocp.CheckpointManager,
    batches_per_epoch: int,
) -> tuple[Transformer, nnx.Optimizer, int]:
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

    # split the abstract model into graphdef, state and rng
    abs_state = nnx.state(abs_model)

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

    total_steps = batches_per_epoch * config.EPOCHS
    warmup_steps = int(0.05 * total_steps)
    lr_schedule_fn = optax.warmup_cosine_decay_schedule(
        init_value=config.INIT_LR,
        peak_value=config.LR,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=config.LR * 0.1,
    )

    # create optimizer
    opt_adamw_with_schedule = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=lr_schedule_fn,
            b1=0.9,
            b2=0.98,
            eps=1e-9,
            weight_decay=config.WEIGHT_DECAY,
        ),
    )

    # create abstract optimizer
    abs_opt = nnx.eval_shape(
        lambda: nnx.Optimizer(abs_model, opt_adamw_with_schedule, wrt=nnx.Param)
    )
    # # get the optimizer state
    abs_opt_state = nnx.state(abs_opt)
    optimizer = nnx.Optimizer(model, opt_adamw_with_schedule, wrt=nnx.Param)

    # restore the state
    latest = manager.latest_step()
    if latest is not None:
        # latest = manager.best_step()
        restored = manager.restore(
            step=latest,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(abs_state),
                optimizer=ocp.args.StandardRestore(abs_opt_state),
                metrics=ocp.args.JsonRestore(),  # optional
            ),
        )
        nnx.update(model, restored["state"])
        nnx.update(optimizer, restored["optimizer"])
        # return the restored model and optimizer
        # assert latest
        return model, optimizer, latest + 1

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

    return model, optimizer, 0
