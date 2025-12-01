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
        rng, epoch_rng = jax.random.split(rng)
        # iterate through each batch in the dataset
        for batch in train_loader.__iter__(rng=epoch_rng):
            epoch_rng, dropout_rng = jax.random.split(epoch_rng)
            # train on batch
            state, train_loss = train_step(
                state=state, batch=batch, dropout_rng=dropout_rng
            )

        # train and val accuracy and loss
        eval_accuracy, eval_loss = eval(state=state, loader=val_loader, rng=epoch_rng)
        train_accuracy, _ = eval(
            state=state, loader=train_loader, rng=epoch_rng, is_train=True
        )

        metrics = {
            "train_loss": float(train_loss),
            "eval_loss": float(eval_loss),
            "train_accuracy": float(train_accuracy),
            "eval_accuracy": float(eval_accuracy),
        }
        # log the metrics
        logger.info(metrics)
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

    src = batch["src"]
    src_mask = batch["src_mask"]
    target_input = batch["target_input"]
    target_output = batch["target_output"]
    target_mask = batch["target_mask"]
    token_mask = batch["token_mask"]

    # define loss function
    def loss_fn(params):
        """
        Compute the loss function for a single batch
        """
        logits = state.apply_fn(
            {"params": params},
            src,
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
        loss = (per_token_loss * token_mask).sum() / token_mask.sum()
        return loss

    # compute loss and gradients
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    # update the the training state with the new gradients
    state = state.apply_gradients(grads=grads)
    return state, loss


def eval(
    state: train_state.TrainState,
    loader,
    rng: jax.random.PRNGKey,
    is_train: bool = False,
) -> tuple[float, float]:
    """
    evaluate the model on the validation set
    Args:
        state: train_state.TrainState
        val_loader: DataLoader

    Returns:
        accuracy, loss
    """
    total_loss = 0.0
    total_accuracy = 0.0
    num_bathces = 0
    # loop over the dataset
    for batch in loader.__iter__(rng=rng):
        # evaluate on batch
        accuracy, loss = eval_step(state=state, batch=batch, is_train=is_train)
        # get num of examples in current batch and add to total
        total_accuracy += accuracy
        total_loss += loss
        num_bathces += 1

    accuracy = total_accuracy / num_bathces
    avg_loss = total_loss / num_bathces

    return accuracy, avg_loss


@jax.jit
def eval_step(state: train_state.TrainState, batch, is_train: bool = False):
    """
    evaluate the model on a single batch
    Args:
        state: train_state.TrainState
        batch: batch

    Returns:
        predictions
    """
    src = batch["src"]
    src_mask = batch["src_mask"]
    target_input = batch["target_input"]
    target_output = batch["target_output"]
    target_mask = batch["target_mask"]
    token_mask = batch["token_mask"]
    # pass batch through the model in training state
    logits = state.apply_fn(
        {"params": state.params},
        src,
        src_mask,
        target_input,
        target_mask,
        is_training=False,
    )
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=target_output,
    )
    loss = (per_token_loss * token_mask).sum() / token_mask.sum()

    predictions = jnp.argmax(logits, axis=-1)
    correct = ((predictions == target_output) * token_mask).sum()
    total = token_mask.sum()
    accuracy = correct / total

    return accuracy, loss
