from flax import nnx
from torch.utils.data import DataLoader

from vision_transformer.model import VisionTransformer


def train(
    model: VisionTransformer,
    train_loader: DataLoader,
    optimizer: nnx.Optimizer,
    num_epochs: int,
):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            continue

    return model
