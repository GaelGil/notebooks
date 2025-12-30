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
    train_loader: grain.DataLoader,
    val_loader: grain.DataLoader,
    epochs: int,
    manager: ocp.CheckpointManager,
    logger: logging,
    batches_per_epoch: int,
    val_batches_per_epoch: int,
    step: int,
):
    """
    Train the model
    """

    current_epoch = step
    batch_in_epoch = 0
    epoch_token_count = 0
    epoch_loss_sum = 0
    # create progress bar
    # total is number of batches per epoch
    # desc is current epoch over total epochs
    pbar = tqdm(
        total=batches_per_epoch,
        desc=f"Epoch {current_epoch}/{epochs}",
    )

    for batch in train_loader:
        try:
            (
                model,
                optimizer,
                avg_batch_loss,
                non_padded_loss,
                num_non_padded_tokens,
            ) = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
            )

        except StopIteration:
            break

        # update current batch in epoch
        batch_in_epoch += 1
        epoch_token_count += (
            num_non_padded_tokens  # number of non padded tokens in the batch
        )
        epoch_loss_sum += non_padded_loss  # loss over non padded tokens

        # update progress bar
        pbar.update(1)
        pbar.set_postfix(loss=f"{jnp.exp(avg_batch_loss):.4f}")

        # check if epoch is complete (the current batch is number of batches per epoch)
        if batch_in_epoch == batches_per_epoch:
            epoch_loss = epoch_loss_sum / epoch_token_count
            logger.info(
                f"Epoch {current_epoch} complete, Avg loss at epoch: {jnp.exp(epoch_loss):.4f}, evaluating ..."
            )
            # evaluate the model
            eval_accuracy, eval_loss = eval(
                model=model, loader=val_loader, batches_per_epoch=val_batches_per_epoch
            )

            # create metrics dictionary
            metrics = {
                "train_perplexity": float(jnp.exp(epoch_loss)),
                "eval_perplexity": float(jnp.exp(eval_loss)),
                "eval_accuracy": float(eval_accuracy),
            }
            # log the metrics
            logger.info(f" EPOCH: {current_epoch} | METRICS: {metrics}, ")
            logger.info(f"Saving checkpoint at epoch {current_epoch}")
            # split the graphdef and state
            _graphdef, state, _rng = nnx.split(
                model,
                nnx.Param,  # trainable weights
                nnx.RngState,
            )
            # save the state, metrics and step
            # opt_state = nnx.state(optimizer)  # optimizer state
            manager.save(
                step=current_epoch,
                args=ocp.args.Composite(
                    state=ocp.args.StandardSave(state),
                    # optimizer=ocp.args.StandardSave(opt_state),
                    metrics=ocp.args.JsonSave(metrics),
                ),
            )
            # update current epoch
            current_epoch += 1
            # if current epoch is equal to total epochs, break
            if current_epoch == epochs:
                break
            # since we have completed an epoch, reset the batch in epoch
            batch_in_epoch = 0
            # reset the epoch loss sum
            epoch_loss_sum = 0
            # reset the epoch token count
            epoch_token_count = 0

            # close the progress bar
            pbar.close()
            # create a new progress bar
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
) -> tuple[Transformer, nnx.Optimizer, float]:
    """
    Train the model on a single batch
    Args:
        model: Transformer
        batch: batch
        optimizer: nnx.Optimizer

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
        )  # loss per token in the batch (B, seq_len)
        num_non_padded_tokens = labels_mask.sum()  # number of non padded tokens
        non_padded_loss = (
            per_token_loss * labels_mask
        ).sum()  # loss over non padded tokens
        loss = non_padded_loss / num_non_padded_tokens
        return (
            loss,
            non_padded_loss,
            num_non_padded_tokens,
        )  # avg loss over batch of non padded tokens, loss over non padded tokens, number of non padded tokens

    # compute loss and gradients
    (loss, non_padded_loss, num_tokens), grads = nnx.value_and_grad(loss_fn)(model)
    # update the model state with the new gradients
    optimizer.update(model, grads)

    return model, optimizer, loss, non_padded_loss, num_tokens


def eval(
    model: Transformer,
    loader: grain.DataLoader,
    batches_per_epoch: int,
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
    current_epoch = 1

    # create a progress bar
    pbar = tqdm(
        total=batches_per_epoch,
        desc=f"Epoch {current_epoch}/{1}",
    )
    for batch in loader:
        try:
            correct_tokens, batch_loss, num_tokens, non_padded_loss = eval_step(
                model=model, batch=batch
            )
            total_correct += correct_tokens
            total_loss += batch_loss
            total_tokens += num_tokens
            pbar.update(1)
            pbar.set_postfix(loss=f"{batch_loss:.4f}")
        except StopIteration:
            break

    # Compute final metrics
    accuracy = total_correct / total_tokens
    avg_cross_entropy = total_loss / total_tokens
    # close the progress bar
    pbar.close()

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

    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    )  # loss per token in the batch (B, seq_len)
    num_non_padded_tokens = labels_mask.sum()  # number of non padded tokens
    non_padded_loss = (
        per_token_loss * labels_mask
    ).sum()  # loss over non padded tokens
    avg_loss_over_batch = (
        non_padded_loss / num_non_padded_tokens
    )  # avg loss over batch of non padded tokens

    # get predictions
    predictions = jnp.argmax(logits, axis=-1)
    # get correct predictions
    correct = predictions == labels
    correct = correct * labels_mask

    return jnp.sum(correct), avg_loss_over_batch, num_non_padded_tokens, non_padded_loss
