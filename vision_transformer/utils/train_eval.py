"""
This training and evaluation file is based on the implementation from
https://wandb.ai/jax-series/simple-training-loop/reports/Writing-a-Training-Loop-in-JAX-and-Flax--VmlldzoyMzA4ODEy
"""

import grain
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from torch.utils.data import DataLoader
from tqdm import tqdm
from flax import nnx
from vision_transformer.VisionTransformer import VisionTransformer
from absl import logging
from jax import Array


def train(
    model: VisionTransformer,
    optimizer: nnx.Optimizer,
    train_loader: grain.DataLoader,
    val_loader: grain.DataLoader,
    epochs: int,
    manager: ocp.CheckpointManager,
    logger: logging,
    step: int,
) -> VisionTransformer:
    """
    Train the model
    Args:
        model: VisionTransformer
        optimizer: nnx.Optimizer
        train_loader: DataLoader
        val_loader: DataLoader
        epochs: int
        manager: ocp.CheckpointManager
        logger: logging
        step: int

    Returns:
        VisionTransformer
    """
    # initialize the random number generator for dropout
    rng = jax.random.PRNGKey(0)
    # loop over the dataset for num_epochs
    for epoch in range(step, epochs):
        # create a tqdm progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in progress_bar:
            rng, dropout_key = jax.random.split(rng)
            step_rngs = nnx.Rngs(dropout=dropout_key)
            # train on batch
            (model, optimizer, _) = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                rngs=step_rngs,
            )

        # get eval/train accuracy and loss
        eval_accuracy, eval_loss = eval(model=model, val_loader=val_loader)
        train_accuracy, train_loss = eval(model=model, val_loader=train_loader)
        progress_bar.set_postfix(
            train_accuracy=train_accuracy,
            eval_accuracy=eval_accuracy,
        )
        # save the metrics
        metrics = {
            "train_loss": float(train_loss),
            "eval_loss": float(eval_loss),
            "train_accuracy": float(train_accuracy),
            "eval_accuracy": float(eval_accuracy),
        }
        # log the metrics
        logger.info(metrics)
        logger.info(f"Saving checkpoint at epoch {epoch}")
        # save the state after each epoch
        _graphdef, state, _rng = nnx.split(
            model,
            nnx.Param,  # trainable weights
            nnx.RngState,  # rng state
        )
        # save the state, metrics and step
        # opt_state = nnx.state(optimizer)  # optimizer state
        manager.save(
            step=epoch,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(state),
                # optimizer=ocp.args.StandardSave(opt_state),
                metrics=ocp.args.JsonSave(metrics),
            ),
        )

    logger.info("Training complete, waiting until all checkpoints are saved")
    manager.wait_until_finished()

    return model


@nnx.jit
def train_step(
    model: VisionTransformer,
    batch,
    optimizer: nnx.Optimizer,
    rngs: jax.Array,
) -> tuple[VisionTransformer, nnx.Optimizer, Array]:
    """
    Handle a single training step. Get loss. Get gradients. Update parameters

    Args:
        model: VisionTransformer
        batch: batch
        dropout_rng: random number generator

    Returns:
        tuple[VisionTransformer, nnx.Optimizer, Any]
    """
    image, label = batch  # unpack the batch

    # define loss function
    def loss_fn(model: VisionTransformer, rngs: nnx.Rngs) -> Array:
        """
        Compute the loss function for a single batch
        """
        # pass batch through the model in training state
        logits = model(
            x=image,
            is_training=True,
            rngs=rngs,
        )
        # calculate mean loss for the batch
        loss: Array = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=label
        ).mean()
        return loss

    (loss), grads = nnx.value_and_grad(
        loss_fn,
        has_aux=True,
    )(model, rngs=rngs)
    optimizer.update(model, grads)

    return model, optimizer, loss


def eval(model: VisionTransformer, loader: DataLoader) -> tuple[Array, Array]:
    """
    Evaluate the model on the eval set
    Args:
        model: VisionTransformer
        loader: DataLoader

    Returns:
        tuple[Array, Array]
    """
    total = 0
    num_correct = 0
    total_loss = 0
    num_batches = 0
    # loop over the dataset
    for batch in loader:
        # evaluate on batch
        correct, loss = eval_step(model=model, batch=batch)
        # get total number of examples
        total += correct.shape[0]
        # get number of correct predictions (will be boolean so we can sum)
        num_correct += correct.sum()
        # get total loss
        total_loss += loss
        num_batches += 1

    # calculate accuracy from number of correct predictions and total samples
    accuracy = num_correct / total
    # calculate average loss from total loss and number of batches
    avg_loss = total_loss / num_batches
    return accuracy, avg_loss


@nnx.jit
def eval_step(model: VisionTransformer, batch) -> tuple[Array, Array]:
    """
    Evaluate the model on a batch
    Args:
        model: VisionTransformer
        batch: batch

    Returns:
        tuple[Array, Array]
    """
    # label shape is (batch_size,)
    image, label = batch  # unpack the batch
    # pass batch through the model in training state
    # logits shape is (batch_size, output_size)
    logits = model(
        x=image,
        is_training=False,
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=label
    ).mean()
    # get predictions from logits using argmax
    preditcions = jnp.argmax(logits, axis=1)
    # get correct predictions
    correct = preditcions == label
    return correct, loss
