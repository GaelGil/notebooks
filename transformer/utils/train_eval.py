import grain
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from absl import logging
from flax import nnx
from tqdm import tqdm
from jax import Array
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
    rng = jax.random.PRNGKey(0)
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
        rng, dropout_key = jax.random.split(rng)
        step_rngs = nnx.Rngs(dropout=dropout_key)
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
                rngs=step_rngs,
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
        pbar.set_postfix(
            loss=f"{jnp.exp(avg_batch_loss):.4f}"
        )  # non padded loss over number of non padded tokens in the batch

        # check if epoch is complete (the current batch is number of batches per epoch)
        if batch_in_epoch == batches_per_epoch:
            # non padded loss over number of non padded tokens for all batches in the epoch
            epoch_loss = epoch_loss_sum / epoch_token_count
            logger.info(
                f"Epoch {current_epoch} complete, Loss at epoch: {jnp.exp(epoch_loss):.4f}, evaluating ..."
            )
            # evaluate the model
            eval_accuracy, eval_loss = eval(
                model=model, loader=val_loader, batches_per_epoch=val_batches_per_epoch
            )

            # create metrics dictionary
            metrics = {
                "train_loss": epoch_loss,
                "eval_loss": eval_loss,
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
                desc=f"Epoch {current_epoch}/{epochs}",
            )

    manager.wait_until_finished()

    return state


@nnx.jit
def train_step(
    model: Transformer,
    batch,
    optimizer: nnx.Optimizer,
    rngs: jax.Array,
) -> tuple[Transformer, nnx.Optimizer, Array, Array, Array]:
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

    # rng, dropout_key = jax.random.split(rng)
    # rngs = nnx.Rngs(dropout=dropout_key)  # <-- step RNG stream

    def loss_fn(model: Transformer, rngs: nnx.Rngs):
        """
        Compute the loss function for a single batch
        """
        # key = jax.random.PRNGKey(0)
        # rngs = nnx.Rngs(dropout=key)
        logits = model(
            src=encoder_input,
            src_mask=encoder_padding_mask,
            target=decoder_input,
            self_mask=decoder_self_attention_mask,
            cross_mask=encoder_decoder_mask,
            is_training=True,
            rngs=rngs,
        )
        per_token_loss: Array = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels
        )  # loss per token in the batch (B, seq_len)
        num_non_padded_tokens: Array = labels_mask.sum()  # number of non padded tokens
        non_padded_loss: Array = (
            per_token_loss * labels_mask
        ).sum()  # loss over non padded tokens
        loss: Array = non_padded_loss / num_non_padded_tokens
        return (
            loss,
            (
                non_padded_loss,
                num_non_padded_tokens,
            ),
        )
        # avg loss over batch of non padded tokens, loss over non padded tokens, number of non padded tokens

    # compute loss and gradients
    (loss, (non_padded_loss, num_tokens)), grads = nnx.value_and_grad(
        loss_fn,
        has_aux=True,
    )(model, rngs=rngs)
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
    total_correct = 0.0  # total number of correct predictions
    eval_loss = 0.0  # total loss
    total_tokens = 0.0  # total number of tokens
    current_epoch = 1  # current epoch

    # create a progress bar
    pbar = tqdm(
        total=batches_per_epoch,
        desc=f"Epoch {current_epoch}/{1}",
    )
    rng_key: jax.Array = jax.random.PRNGKey(0)
    for batch in loader:
        try:
            key, subkey = jax.random.split(rng_key)
            correct_tokens, batch_loss, num_tokens, non_padded_loss = eval_step(
                model=model, batch=batch, rng=subkey
            )
            total_correct += correct_tokens
            eval_loss += non_padded_loss
            total_tokens += num_tokens
            pbar.update(1)
            pbar.set_postfix(loss=f"{batch_loss:.4f}")
        except StopIteration:
            break

    # Compute final metrics
    accuracy = total_correct / total_tokens
    eval_loss = eval_loss / total_tokens
    # close the progress bar
    pbar.close()

    return accuracy, eval_loss


@nnx.jit
def eval_step(
    model: Transformer,
    batch,
    rng: jax.Array,
) -> tuple[Array, Array, Array, Array]:
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

    # pass batch through the model in eval mode
    logits = model(
        src=encoder_input,
        src_mask=encoder_padding_mask,
        target=decoder_input,
        self_mask=decoder_self_attention_mask,
        cross_mask=encoder_decoder_mask,
        is_training=False,
    )

    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    )  # loss per token in the batch (B, seq_len)
    num_non_padded_tokens: Array = labels_mask.sum()  # number of non padded tokens
    non_padded_loss: Array = (
        per_token_loss * labels_mask
    ).sum()  # loss over non padded tokens
    avg_loss_over_batch: Array = (
        non_padded_loss / num_non_padded_tokens
    )  # avg loss over batch of non padded tokens

    # get predictions
    predictions = jnp.argmax(logits, axis=-1)
    # get correct predictions
    correct = predictions == labels
    # mask out the padded tokens
    correct = correct * labels_mask

    return jnp.sum(correct), avg_loss_over_batch, num_non_padded_tokens, non_padded_loss
