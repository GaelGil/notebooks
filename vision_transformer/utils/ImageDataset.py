import os

import jax.numpy as jnp
import torch
from jax.tree_util import tree_map
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import ImageFolder


class ImageDataset:
    def __init__(self, dataset_path: str, transformations=None) -> None:
        self.dataset = ImageFolder(root=dataset_path, transform=transformations)
        self.dataset_len: int = len(self.dataset)
        self.train_loader: DataLoader
        self.test_loader: DataLoader
        self.val_loader: DataLoader

    def get_length(self) -> int:
        """
        Returns the length of the dataset

        Args:
            None

        Returns:
            int: length of the dataset
        """
        return self.dataset_len

    def numpy_collate(self, batch):
        """
        Collate function for DataLoader

        Args:
            batch: batch

        Returns:
            batch: batch
        """
        return tree_map(jnp.array, default_collate(batch))

    def split_data(
        self,
        train_split: float,
        val_split: float,
        batch_size: int,
        num_workers: int,
        save_splits_path: str,
        load_splits: bool = False,
        seed: int = 42,
    ):
        """
        Split the dataset into train, validation and test sets

        Args:
            train_split: float
            val_split: float
            batch_size: int
            num_workers: int
            save_splits_path: str
            load_splits: bool
            seed: int

        Returns:
            None
        """
        if load_splits and os.path.exists(save_splits_path):
            splits = torch.load(save_splits_path)
        else:
            generator = torch.Generator().manual_seed(seed)
            train_count = int(train_split * self.dataset_len)
            val_count = int(val_split * self.dataset_len)
            test_count = self.dataset_len - train_count - val_count
            train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
                self.dataset, (train_count, val_count, test_count), generator=generator
            )
            splits = {
                "train_indices": train_dataset.indices,
                "val_indices": valid_dataset.indices,
                "test_indices": test_dataset.indices,
            }

            torch.save(splits, save_splits_path)

        # Create subsets from saved indices
        train_dataset = torch.utils.data.Subset(self.dataset, splits["train_indices"])
        valid_dataset = torch.utils.data.Subset(self.dataset, splits["val_indices"])
        test_dataset = torch.utils.data.Subset(self.dataset, splits["test_indices"])

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
            shuffle=False,
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

    def get_loaders(self):
        """
        Returns the train, val and test loaders

        Args:
            None

        Returns:
            train_loader: train loader
            val_loader: val loader
            test_loader: test loader
        """
        return self.train_loader, self.val_loader, self.test_loader
