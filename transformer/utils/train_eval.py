import grain
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from absl import logging
from flax import nnx
from tqdm import tqdm

from transformer.Transformer import Transformer


def train(
    model: Transformer,
    optimizer: nnx.Optimizer,
    train_loader: grain.DataLoaderIterator,
    val_loader: grain.DataLoaderIterator,
    epochs: int,
    manager: ocp.CheckpointManager,
    logger: logging,
    batches_per_epoch: int,
    total_batches: int,
):
    """
    Train the model
    """

    loader_rng = jax.random.PRNGKey(0)

    current_epoch = 0
    batch_in_epoch = 0
    # epoch_losses = []

    pbar = tqdm(
        total=batches_per_epoch,
        desc=f"Epoch {current_epoch}/{epochs}",
    )

    for batch in train_loader:
        try:
            # batch = next(train_loader)
            model, optimizer, batch_loss = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                dropout_rng=loader_rng,
            )
        except StopIteration:
            break

        batch_in_epoch += 1

        # ----- progress -----
        pbar.update(1)
        pbar.set_postfix(loss=f"{batch_loss:.4f}")

        # update progress bar
        if batch_in_epoch == batches_per_epoch:
            logger.info(
                f"Epoch {current_epoch} complete, loss: {batch_loss}, evaluating ..."
            )
            # evaluate the model
            eval_accuracy, eval_loss = eval(model=model, loader=val_loader)
            # train_accuracy, train_loss = eval(model=model, loader=train_loader)

            # create metrics dictionary
            metrics = {
                # "train_perplexity": float(jnp.exp(train_loss)),
                "eval_perplexity": float(jnp.exp(eval_loss)),
                # "train_accuracy": float(train_accuracy),
                "eval_accuracy": float(eval_accuracy),
            }
            # log the metrics
            logger.info(f" EPOCH: {current_epoch} | METRICS: {metrics}, ")
            logger.info(f"Saving checkpoint at epoch {current_epoch}")
            # split the graphdef and state
            _, state = nnx.split(model, nnx.Param, nnx.State)
            # save the state, optimizer and metrics and step
            manager.save(
                step=current_epoch,
                args=ocp.args.Composite(
                    state=ocp.args.StandardSave(state),
                    optimizer=ocp.args.StandardSave(optimizer),
                    metrics=ocp.args.JsonSave(metrics),
                ),
            )
            current_epoch += 1
            if current_epoch == epochs:
                break

            batch_in_epoch = 0
            # epoch_losses.clear()

            pbar.close()
            pbar = tqdm(
                total=batches_per_epoch,
                desc=f"Epoch {current_epoch + 1}/{epochs}",
            )

    manager.wait_until_finished()

    return state


@jax.jit
def train_step(
    model: Transformer,
    batch,
    optimizer: nnx.Optimizer,
    dropout_rng: jax.random.PRNGKey,
) -> tuple[Transformer, nnx.Optimizer, float]:
    """
    Train the model on a single batch
    Args:
        model: Transformer
        batch: batch
        optimizer: nnx.Optimizer
        dropout_rng: jax.random.PRNGKey

    Returns:
        state, loss
    """

    (
        encoder_input,
        decoder_input,
        labels,
        labels_mask,
        encoder_padding_mask,
        decoder_self_attention_mask,
        encoder_decoder_mask,
    ) = batch

    # define loss function
    def loss_fn(model: Transformer):
        """
        Compute the loss function for a single batch
        """
        key = jax.random.PRNGKey(0)

        rngs = nnx.Rngs(dropout=key)
        logits = model(
            src=encoder_input,
            src_mask=encoder_padding_mask,
            target=decoder_input,
            self_mask=decoder_self_attention_mask,
            cross_mask=encoder_decoder_mask,
            is_training=True,
            rngs=rngs,
        )
        per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels
        )
        loss = (per_token_loss * labels_mask).sum() / labels_mask.sum()
        return loss

    # compute loss and gradients
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    # update the the training state with the new gradients
    optimizer.update(model, grads)

    return model, optimizer, loss


def eval(
    model: Transformer,
    loader,
) -> tuple[float, float]:
    """
    evaluate the model on the validation set
    Args:
        state: train_state.TrainState
        val_loader: DataLoader

    Returns:
        accuracy, loss
    """
    total_correct = 0.0
    total_loss = 0.0
    total_tokens = 0.0
    # loop over the dataset
    for batch in loader:
        try:
            # batch = next(loader)
            correct_tokens, batch_loss, num_tokens = eval_step(model=model, batch=batch)
            total_correct += correct_tokens
            total_loss += batch_loss
            total_tokens += num_tokens
            # if num_tokens == 0:
            #     print("num tokens is 0")
            # else:
            #     print(f"num tokens: {num_tokens}")
        except StopIteration:
            break

    # print(f"total tokens: {total_tokens}")
    # Compute final metrics
    accuracy = total_correct / total_tokens
    avg_cross_entropy = total_loss / total_tokens
    # perplexity = jnp.exp(avg_cross_entropy)

    return float(accuracy), float(avg_cross_entropy)


@jax.jit
def eval_step(
    model: Transformer, batch
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    evaluate the model on a single batch
    Args:
        state: train_state.TrainState
        batch: batch

    Returns:
        predictions
    """
    (
        encoder_input,
        decoder_input,
        labels,
        labels_mask,
        encoder_padding_mask,
        decoder_self_attention_mask,
        encoder_decoder_mask,
    ) = batch

    # pass batch through the model in training state
    key = jax.random.PRNGKey(0)

    rngs = nnx.Rngs(dropout=key)
    logits = model(
        src=encoder_input,
        src_mask=encoder_padding_mask,
        target=decoder_input,
        self_mask=decoder_self_attention_mask,
        cross_mask=encoder_decoder_mask,
        is_training=False,
        rngs=rngs,
    )

    # compute loss
    cross_entropy_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    )

    loss = (cross_entropy_loss * labels_mask).sum() / labels_mask.sum()

    predictions = jnp.argmax(logits, axis=-1)

    correct = predictions == labels
    correct = correct * labels_mask

    # accuracy = jnp.sum(correct) / jnp.sum(labels_mask)

    return jnp.sum(correct), loss, jnp.sum(labels_mask)
