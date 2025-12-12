import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from transformer.Transformer import Transformer
from flax import nnx
import grain


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

    rng = jax.random.PRNGKey(0)
    for epoch in range(step, epochs):
        rng, loader_rng = jax.random.split(rng)
        for batch in train_loader.__iter__():
            batch = next(train_loader)
            model, optimizer, batch_loss = train_step(
                model=model, batch=batch, optimizer=optimizer, dropout_rng=loader_rng
            )

        # after each batch evaluate
        eval_accuracy, eval_loss = eval(model=model, loader=val_loader, rng=None)
        train_accuracy, train_loss = eval(
            model=model, loader=train_loader, rng=loader_rng
        )

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

    x = batch["src"]
    y = batch["target"]

    # define loss function
    def loss_fn(params):
        """
        Compute the loss function for a single batch
        """
        logits = model(x, params, dropout_rng)
        loss = optax.softmax_cross_entropy(logits=logits, labels=y)
        return loss

    # compute loss and gradients
    loss, grads = nnx.value_and_grad(loss_fn)(model.params)
    # update the the training state with the new gradients
    optimizer.update(model, grads)

    return model, optimizer, loss


def eval(
    model: Transformer,
    loader,
    optimizer: nnx.Optimizer,
    dropout_rng: jax.random.PRNGKey,
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
    for batch in loader.__iter__(rng=None):
        correct_tokens, batch_loss, num_tokens = eval_step(moedl=model, batch=batch)
        total_correct += correct_tokens
        total_loss += batch_loss
        total_tokens += num_tokens

    # Compute final metrics
    accuracy = total_correct / total_tokens
    avg_cross_entropy = total_loss / total_tokens
    # perplexity = jnp.exp(avg_cross_entropy)

    return float(accuracy), float(avg_cross_entropy)


@jax.jit
def eval_step(model: Transformer, batch) -> tuple[float, float, float]:
    """
    evaluate the model on a single batch
    Args:
        state: train_state.TrainState
        batch: batch

    Returns:
        predictions
    """
    x = batch["src"]
    y = batch["target"]

    # pass batch through the model in training state
    logits = model(x)

    # compute loss
    cross_entropy_loss = optax.softmax_cross_entropy(logits=logits, labels=y)

    # compute accuracy
    correct_tokens = jnp.sum(jnp.argmax(logits, axis=-1) == jnp.argmax(y, axis=-1))
    num_tokens = jnp.size(correct_tokens)

    return correct_tokens, cross_entropy_loss, num_tokens
