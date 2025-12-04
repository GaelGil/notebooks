"""
This training and evaluation file is based on the implementation from
https://wandb.ai/jax-series/simple-training-loop/reports/Writing-a-Training-Loop-in-JAX-and-Flax--VmlldzoyMzA4ODEy
"""

from typing import Any

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax.training import train_state


def train(
    state: train_state.TrainState,
    train_loader,
    val_loader,
    epochs: int,
    manager: ocp.CheckpointManager,
    logger,
    step: int = 0,
):
    """
    train the model
    Args:
        state: train_state.TrainState
        train_loader: DataLoader
        val_loader: DataLoader
        epochs: int
        manager: ocp.CheckpointManager
        logger: logger

    Returns:
        None
    """

    rng = jax.random.PRNGKey(0)
    # loop over the dataset for num_epochs
    for epoch in range(step, epochs):
        # split for this epoch once
        rng, loader_rng = jax.random.split(rng)

        # split again for dropout
        rng, dropout_base = jax.random.split(rng)
        # iterate through each batch in the dataset
        for batch in train_loader.__iter__(rng=loader_rng):
            dropout_base, dropout_rng = jax.random.split(dropout_base)
            # train on batch
            state, loss = train_step(state=state, batch=batch, dropout_rng=dropout_rng)

        # train and val accuracy and loss
        eval_accuracy, eval_loss = eval(state=state, loader=val_loader, rng=None)
        train_accuracy, train_loss = eval(
            state=state, loader=train_loader, rng=loader_rng
        )

        # create metrics dictionary
        metrics = {
            "train_perplexity": float(train_loss),
            "eval_perplexity": float(eval_loss),
            "train_accuracy": float(train_accuracy),
            "eval_accuracy": float(eval_accuracy),
        }
        # log the metrics
        logger.info(f" EPOCH: {epoch} | METRICS: {metrics}")
        logger.info(f"Saving checkpoint at epoch {epoch}")
        manager.save(
            step=epoch,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(state),
                metrics=ocp.args.JsonSave(metrics),
            ),
        )

    manager.wait_until_finished()

    return state


@jax.jit
def train_step(
    state: train_state.TrainState,
    batch,
    dropout_rng: jax.random.PRNGKey,
) -> tuple[train_state.TrainState, Any]:
    """
    handle a single training step
    get loss
    get gradients
    update parameters

    Args:
        state: train_state.TrainState
        batch: batch
        dropout_rng: random number generator

    Returns:
        train_state.TrainState and loss
    """

    src_input = batch["src_input"]
    src_mask = batch["src_mask"]
    target_input = batch["target_input"]
    target_mask = batch["target_mask"]
    target_output = batch["target_output"]
    target_output_mask = batch["target_output_mask"]

    # define loss function
    def loss_fn(params):
        """
        Compute the loss function for a single batch
        """
        logits = state.apply_fn(
            {"params": params},
            src_input,
            src_mask,
            target_input,
            target_mask,
            is_training=True,
            rngs={"dropout": dropout_rng},
        )

        per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=target_output,
        )
        masked_loss = per_token_loss * target_output_mask
        cross_entropy_loss = masked_loss.sum() / target_output_mask.sum()
        return cross_entropy_loss

    # compute loss and gradients
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    # update the the training state with the new gradients
    state = state.apply_gradients(grads=grads)
    return state, jnp.exp(loss)


def eval(
    state: train_state.TrainState,
    loader,
    rng: jax.random.PRNGKey,
) -> tuple[float, float]:
    """
    evaluate the model on the validation set
    Args:
        state: train_state.TrainState
        val_loader: DataLoader

    Returns:
        accuracy, loss
    """
    total_perplexity = 0.0
    total_accuracy = 0.0
    num_bathces = 0
    # loop over the dataset
    for batch in loader.__iter__(rng=rng):
        # evaluate on batch
        accuracy, perplexity = eval_step(state=state, batch=batch)
        # get num of examples in current batch and add to total
        total_accuracy += accuracy
        total_perplexity += perplexity
        num_bathces += 1

    accuracy = total_accuracy / num_bathces
    avg_loss = total_perplexity / num_bathces

    return accuracy, avg_loss


@jax.jit
def eval_step(state: train_state.TrainState, batch):
    """
    evaluate the model on a single batch
    Args:
        state: train_state.TrainState
        batch: batch

    Returns:
        predictions
    """
    src_input = batch["src_input"]
    src_mask = batch["src_mask"]
    target_input = batch["target_input"]
    target_mask = batch["target_mask"]
    target_output = batch["target_output"]
    target_output_mask = batch["target_output_mask"]
    # pass batch through the model in training state
    logits = state.apply_fn(
        {"params": state.params},
        src_input,
        src_mask,
        target_input,
        target_mask,
        is_training=False,
    )
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=target_output,
    )
    masked_loss = per_token_loss * target_output_mask
    cross_entropy_loss = masked_loss.sum() / target_output_mask.sum()
    perplexity = jnp.exp(cross_entropy_loss)
    # logits shape (B, L, vocab)
    pred = jnp.argmax(logits, axis=-1)  # (B, L)
    # mask = target_output_mask (B, L) with 1 for real tokens, 0 for padding
    correct = jnp.sum((pred == target_output) * target_output_mask)
    total = jnp.sum(target_output_mask)
    accuracy = correct / total  # -> in [0,1]

    return accuracy, perplexity
