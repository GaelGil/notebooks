import jax.numpy as jnp
import optax
from flax import nnx
from torch.utils.data import DataLoader
from tqdm import tqdm

from vision_transformer.model import VisionTransformer


def train(
    model: VisionTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: nnx.Optimizer,
    num_epochs: int,
):
    model.train()
    # loop over the dataset for num_epochs
    for epoch in range(num_epochs):
        # create a tqdm progress bar
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False
        )
        # iterate through each batch in the dataset
        for batch in train_loader:
            # train on batch
            train_step(model=model, optimizer=optimizer, batch=batch)

        # after each epoch, evaluate on train and val set
        progress_bar.set_postfix(
            train_accuracy=eval(model=model, val_loader=train_loader),
            eval_accuracy=eval(model=model, val_loader=val_loader),
        )


@nnx.jit
def train_step(model: VisionTransformer, optimizer: nnx.Optimizer, batch):
    """
    Training step as implemented in
    "https://cloud.google.com/blog/products/ai-machine-learning/guide-to-jax-for-pytorch-developers"

    Args:
        model: model
        optimizer: optimizer
        batch: batch

    Returns:
        None
    """

    # define loss function
    def loss_fn(model):
        #
        logits = model(batch[0])
        # calculate mean loss for the batch
        loss = optax.softmax_cross_entropy(logits=logits, labels=batch[1]).mean()
        return loss

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model)
    # update the parameters
    optimizer.update(grads)


def eval(model: VisionTransformer, val_loader: DataLoader):
    # set model to eval mode
    model.eval()
    total = 0
    num_correct = 0
    # loop over the dataset
    for batch in val_loader:
        # evaluate on batch
        res = eval_step(model, batch)
        # get total number of examples
        total += res.shape[0]
        # get number of correct predictions (will be boolean so we can sum)
        num_correct += res.sum()

    return num_correct / total


@nnx.jit
def eval_step(model: VisionTransformer, batch):
    logits = model(batch[0])
    logits = logits.squeeze()
    preditcions = jnp.round(nnx.softmax(logits))
    return preditcions == batch[1]
