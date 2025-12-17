import grain
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx

from transformer.Transformer import Transformer


def train(
    model: Transformer,
    optimizer: nnx.Optimizer,
    train_loader: grain.DataLoaderIterator,
    val_loader: grain.DataLoaderIterator,
    epochs: int,
    manager: ocp.CheckpointManager,
    logger,
    tokenizer,
    step: int = 0,
):
    """
    Train the model
    """

    loader_rng = jax.random.PRNGKey(0)
    # train_accuracy, train_loss = eval(model=model, loader=val_loader)

    # print(f"train perplexity: {jnp.exp(train_loss)} | train accuracy: {train_accuracy}")
    rng = jax.random.PRNGKey(0)
    for epoch in range(step, epochs):
        rng, loader_rng = jax.random.split(rng)
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

        # after each batch evaluate
        train_accuracy, train_loss = eval(model=model, loader=train_loader)
        eval_accuracy, eval_loss = eval(model=model, loader=val_loader)

        # create metrics dictionary
        metrics = {
            "train_perplexity": float(jnp.exp(train_loss)),
            "eval_perplexity": float(jnp.exp(eval_loss)),
            "train_accuracy": float(train_accuracy),
            "eval_accuracy": float(eval_accuracy),
        }
        # log the metrics
        logger.info(f" EPOCH: {epoch} | METRICS: {metrics}")
        logger.info(f"Saving checkpoint at epoch {epoch}")
        _, state = nnx.split(model, nnx.Param, nnx.State)
        manager.save(
            step=epoch,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(state),
                optimizer=ocp.args.StandardSave(optimizer),
                metrics=ocp.args.JsonSave(metrics),
            ),
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
            total_correct += correct_tokens.item()
            total_loss += batch_loss.item()
            total_tokens += num_tokens.item()
        except StopIteration:
            break

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
        _,
        encoder_padding_mask,
        decoder_self_attention_mask,
        encoder_decoder_mask,
    ) = batch

    labels_mask = labels != 0  # ✅ CORRECT loss mask
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

    loss_sum = jnp.sum(cross_entropy_loss * labels_mask)
    num_tokens = jnp.sum(labels_mask)

    preds = jnp.argmax(logits, axis=-1)
    correct_tokens = jnp.sum((preds == labels) & labels_mask)

    return correct_tokens, loss_sum, num_tokens
