"""
This training and evaluation file is based on the implementation from
https://wandb.ai/jax-series/simple-training-loop/reports/Writing-a-Training-Loop-in-JAX-and-Flax--VmlldzoyMzA4ODEy
"""

from typing import Any

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import linen as nn
from flax.training import train_state
from torch.utils.data import DataLoader
from tqdm import tqdm

from vision_transformer.model import VisionTransformer


def train(
    state: train_state.TrainState,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    manager: ocp.CheckpointManager,
    logger,
):
    # initialize the random number generator for dropout
    rng = jax.random.PRNGKey(0)
    # loop over the dataset for num_epochs
    for epoch in range(epochs):
        # create a tqdm progress bar
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False
        )
        # iterate through each batch in the dataset
        for batch in train_loader:
            # train on batch
            state, _ = train_step(state=state, batch=batch, dropout_rng=rng)

        # after each epoch, evaluate on train and val set
        progress_bar.set_postfix(
            train_accuracy=eval(val_loader=train_loader),
            eval_accuracy=eval(val_loader=val_loader),
        )
        logger.info(f"Saving checkpoint at epoch {epoch}")

        # save the state after each epoch
        manager.save(
            step=epoch,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(state),
            ),
        )

    logger.info("Training complete, waiting until all checkpoints are saved")
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
    image, label = batch # unpack the batch 
    # define loss function
    def loss_fn(params):
        """
        Compute the loss function for a single batch
        """
        # pass batch through the model in training state
        logits = state.apply_fn( {'params': params}, image, rngs={'dropout': dropout_rng})
        # calculate mean loss for the batch
        loss = optax.softmax_cross_entropy(
            logits=logits.squeeze(), labels=label
        ).mean()
        return loss

    # compute loss and gradients
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    # update the the training state with the new gradients
    state = state.apply_gradients(grads=grads)
    return state, loss


def eval(state: train_state.TrainState, val_loader: DataLoader) -> float:
    """
    evaluate the model on the validation set
    Args:
        state: train_state.TrainState
        val_loader: DataLoader

    Returns:
        accuracy
    """
    total = 0
    num_correct = 0
    # loop over the dataset
    for batch in val_loader:
        # evaluate on batch
        res = eval_step(state=state, batch=batch)
        # get total number of examples
        total += res.shape[0]
        # get number of correct predictions (will be boolean so we can sum)
        num_correct += res.sum()

    return num_correct / total


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
    # pass batch through the model in training state
    logits = state.apply_fn(state.params, batch[0])
    logits = logits.squeeze()
    # get predictions from logits
    preditcions = jnp.round(nn.softmax(logits))
    # return number of correct predictions
    return preditcions == batch[1]
