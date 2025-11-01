import jax.numpy as jnp
import torch
from datasets import load_dataset
from jax.tree_util import tree_map
from torch.utils.data import DataLoader, default_collate


class LangDataset:
    def __init__(self, dataset_name: str, src_lang: str, target_lang: str):
        """
        Load the dataset from the Hugging Face Hub
        Args:
            dataset_name: name of the dataset
            src_lang: source language
            target_lang: target language

        Returns:
            None
        """
        self.dataset: dict = load_dataset(dataset_name)
        self.src_lang: str = src_lang
        self.target_lang: str = target_lang
        self.src_data = None
        self.target_data = None
        self.src_train_loader = None
        self.src_val_loader = None
        self.src_test_loader = None
        self.target_train_loader = None
        self.target_val_loader = None
        self.target_test_loader = None

    def __len__(self):
        """
        Get the length of the dataset
        Args:
            None

        Returns:
            None
        """
        return len(self.dataset["train"])

    def __getitem__(self, idx: int):
        src = torch.tensor(self.src_ids[idx]["ids"], dtype=torch.long)
        tgt = torch.tensor(self.target_ids[idx]["ids"], dtype=torch.long)
        return src, tgt

    def get_dataset(self):
        """
        Load the dataset from the Hugging Face Hub
        Args:
            None

        Returns:
            None
        """
        return self.dataset

    def set_src_target_ids(self, src_data: dict, target_data: dict):
        self.src_data = src_data
        self.target_data = target_data

    def valid_pair(self, example):
        src = example.get(self.src_lang)
        tgt = example.get(self.target_lang)

        # Reject if missing or None
        if src is None or tgt is None:
            return False

        # Reject if not strings
        if not isinstance(src, str) or not isinstance(tgt, str):
            return False

        # Reject empty or whitespace
        if not src.strip() or not tgt.strip():
            return False

        return True

    def handle_null(self):
        """
        Check if any examples in the dataset are null.
        If so, remove them and update the dataset with the remaining examples.

        Args:
            None

        Returns:
            None
        """
        self.dataset = self.dataset.filter(self.valid_pair)

    def numpy_collate(self, batch):
        return tree_map(jnp.array, default_collate(batch))

    def split_counts(self):
        src_count = len(self.src_ids)
        target_count = len(self.target_ids)
        return src_count, target_count

    def split(
        self, train_split: float, val_split: float, batch_size: int, seed: int = 42
    ):
        generator = torch.Generator().manual_seed(seed)

        data_count = len(self.src_data)
        train_count = int(train_split * data_count)
        val_count = int(val_split * data_count)
        test_count = data_count - train_count - val_count

        src_train, src_valid, src_test = torch.utils.data.random_split(
            self.src_data,
            (train_count, val_count, test_count),
            generator=generator,
        )

        target_train, target_valid, target_test = torch.utils.data.random_split(
            self.target_data,
            (train_count, val_count, test_count),
            generator=generator,
        )

        src_train_loader = DataLoader(
            src_train,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.numpy_collate,
        )
        src_val_loader = DataLoader(
            src_valid,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.numpy_collate,
        )
        src_test_loader = DataLoader(
            src_test,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.numpy_collate,
        )

        target_train_loader = DataLoader(
            target_train,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.numpy_collate,
        )
        target_val_loader = DataLoader(
            target_valid,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.numpy_collate,
        )
        target_test_loader = DataLoader(
            target_test,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.numpy_collate,
        )
        self.src_train_loader = src_train_loader
        self.src_val_loader = src_val_loader
        self.src_test_loader = src_test_loader
        self.target_train_loader = target_train_loader
        self.target_val_loader = target_val_loader
        self.target_test_loader = target_test_loader

    def get_loaders(self):
        return (
            self.src_train_loader,
            self.src_val_loader,
            self.src_test_loader,
            self.target_train_loader,
            self.target_val_loader,
            self.target_test_loader,
        )
