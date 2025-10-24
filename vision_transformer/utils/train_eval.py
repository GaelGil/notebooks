import optax
from flax import nnx
from torch.utils.data import DataLoader

from vision_transformer.model import VisionTransformer


def train(
    model: VisionTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: nnx.Optimizer,
    num_epochs: int,
):
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            train_step(model=model, optimizer=optimizer, batch=batch)

    return model


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

    def loss_fn(model):
        logits = model(batch[0])
        loss = optax.softmax_cross_entropy(logits=logits, labels=batch[1]).mean()
        return loss

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model)
    optimizer.update(grads)
