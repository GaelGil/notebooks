import jax.numpy as jnp
import torch
from jax.tree_util import tree_map
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import ImageFolder


class ImageDataset:
    def __init__(self, dataset_path: str, transformations=None) -> None:
        self.dataset = ImageFolder(root=dataset_path, transform=transformations)
        self.dataset_len: int
        self.train_loader: DataLoader
        self.test_loader: DataLoader
        self.val_loader: DataLoader

    def get_datset_length(self):
        return self.dataset_len

    def numpy_collate(self, batch):
        return tree_map(jnp.array, default_collate(batch))

    def split_data(
        self, train_split: float, val_split: float, batch_size: int, num_workers: int
    ):
        train_count = int(train_split * self.dataset_len)
        val_count = int(val_split * self.dataset_len)
        test_count = self.dataset_len - train_count - val_count
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
            self.dataset, (train_count, val_count, test_count)
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.numpy_collate,
        )
        self.val_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.numpy_collate,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.numpy_collate,
        )

    def get_data_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader
