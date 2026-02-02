import grain
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageDataset:
    def __init__(self, dataset_path: str, transformations: transforms.Compose) -> None:
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

    def load_splits(self, save_splits_path):
        splits = torch.load(save_splits_path)
        train_dataset = torch.utils.data.Subset(self.dataset, splits["train_indices"])
        val_dataset = torch.utils.data.Subset(self.dataset, splits["val_indices"])
        test_dataset = torch.utils.data.Subset(self.dataset, splits["test_indices"])
        return train_dataset, val_dataset, test_dataset

    def get_loder(
        self,
        dataset,
        seed: int,
        batch_size: int,
        drop_remainder: bool,
        num_workers: int,
    ):
        loader = grain.load(
            dataset,
            shuffle=True,
            seed=seed,
            batch_size=batch_size,
            drop_remainder=drop_remainder,
            worker_count=num_workers,
        )

        return loader

    def split_data(
        self,
        train_split: float,
        val_split: float,
        batch_size: int,
        num_workers: int,
        save_splits_path: str,
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

        generator = torch.Generator().manual_seed(seed)
        train_count = int(train_split * self.dataset_len)
        val_count = int(val_split * self.dataset_len)
        test_count = self.dataset_len - train_count - val_count
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            self.dataset, (train_count, val_count, test_count), generator=generator
        )
        splits = {
            "train_indices": train_dataset.indices,
            "val_indices": val_dataset.indices,
            "test_indices": test_dataset.indices,
        }

        torch.save(splits, save_splits_path)

        return train_dataset, val_dataset, test_dataset
