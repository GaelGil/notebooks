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
    train_batches,
    val_batches,
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
        # iterate through each batch in the dataset
        for batch in train_batches:
            rng, dropout_rng = jax.random.split(rng)
            # train on batch
            state, _ = train_step(state=state, batch=batch, dropout_rng=rng)

        # train and val accuracy and loss
        eval_accuracy, eval_loss = eval(state=state, loader=val_batches)
        train_accuracy, train_loss = eval(state=state, loader=train_batches)

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
    target_mask = batch["target_mask"]

    # define loss function
    def loss_fn(params):
        """
        Compute the loss function for a single batch
        """
        # pass batch through the model in training state
        logits = state.apply_fn(
            {"params": params},
            src,
            src_mask,
            target_input,
            target_mask,
            is_training=True,
            rngs={"dropout": dropout_rng},
        )
        # calculate mean loss for the batch
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=target_input
        ).mean()
        return loss

    # compute loss and gradients
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    # update the the training state with the new gradients
    state = state.apply_gradients(grads=grads)
    return state, loss


def eval(state: train_state.TrainState, val):
    """
    evaluate the model on the validation set
    Args:
        state: train_state.TrainState
        val_loader: DataLoader

    Returns:
        accuracy
    """
    total_batch = 0
    num_correct_batch = 0
    total_loss = 0
    num_bathces = 0
    # loop over the dataset
    for batch in val:
        # evaluate on batch
        res, loss = eval_step(state=state, batch=batch)
        # get num of examples in current batch and add to total
        total_batch += res.shape[0]
        # get number of correct predictions for current batch (will be boolean so we can sum)
        num_correct_batch += res.sum()
        total_loss += loss
        num_bathces += 1

    accuracy = num_correct_batch / total_batch
    avg_loss = total_loss / num_bathces

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
    src = (batch["src"],)
    src_mask = (batch["src_mask"],)
    target_input = (batch["target_input"],)
    target_mask = (batch["target_mask"],)
    # pass batch through the model in training state
    logits = state.apply_fn(
        {"params": state.params},
        src,
        src_mask,
        target_input,
        target_mask,
        is_training=False,
        rngs={"dropout": jax.random.PRNGKey(0)},
    )
    # calculate mean loss for the batch
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=target_input
    ).mean()
    # get predictions from logits using argmax
    preditcions = jnp.argmax(logits, axis=-1)
    # check if predictions are correct
    correct = preditcions == target_input
    return correct, loss
