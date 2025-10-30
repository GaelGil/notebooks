"""
This training and evaluation file is based on the implementation from
"https://cloud.google.com/blog/products/ai-machine-learning/guide-to-jax-for-pytorch-developers"
"""

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx
from flax.training import train_state
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformer.model import Transformer


def train(
    state: train_state.TrainState,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    manager: ocp.CheckpointManager,
):
    # loop over the dataset for num_epochs
    for epoch in range(num_epochs):
        # create a tqdm progress bar
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False
        )
        # iterate through each batch in the dataset
        for batch in train_loader:
            # train on batch

            state, loss = (train_step(model=model, state=state, batch=batch),)

        # after each epoch, evaluate on train and val set
        progress_bar.set_postfix(
            train_accuracy=eval(model=model, val_loader=train_loader),
            eval_accuracy=eval(model=model, val_loader=val_loader),
        )
        # save the state after each epoch
        manager.save(step=epoch, args=ocp.args.StandardSave(state))

    return state


@nnx.jit
def train_step(
    state: train_state.TrainState,
    batch,
):
    """
    handle a single training step
    get loss
    get gradients
    update parameters

    Args:
        model: model
        optimizer: optimizer
        batch: batch

    Returns:
        None
    """

    # define loss function
    def loss_fn(params):
        """
        Compute the loss function for a single batch
        """
        # pass batch through the model in training state
        logits = state.apply_fn(params, batch[0], training=True)
        # calculate mean loss for the batch
        loss = optax.softmax_cross_entropy(
            logits=logits.squeeze(), labels=batch[1]
        ).mean()
        return loss

    # compute loss and gradients
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    # update the the training state with the new gradients
    state = state.apply_gradients(grads=grads)
    return state, loss


def eval(state: train_state.TrainState, val_loader: DataLoader):
    # set model to eval mode
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


@nnx.jit
def eval_step(state: train_state.TrainState, batch):
    # pass batch through the model in training state
    logits = state.apply_fn(state.params, batch[0], training=False)
    logits = logits.squeeze()
    # get predictions from logits
    preditcions = jnp.round(nnx.softmax(logits))
    # return number of correct predictions
    return preditcions == batch[1]
