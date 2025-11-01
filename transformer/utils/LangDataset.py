import jax.numpy as jnp
import torch
from datasets import load_dataset
from jax.tree_util import tree_map
from torch.utils.data import DataLoader, default_collate


class LangDataset:
    def __init__(self, dataset_name: str, src_lang: str, target_lang: str):
        self.dataset: dict = load_dataset(dataset_name)
        self.src_lang: str = src_lang
        self.target_lang: str = target_lang
        self.src_ids = None
        self.target_ids = None

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

    def set_src_target_ids(self, src_ids: dict, target_ids: dict):
        self.src_ids = src_ids
        self.target_ids = target_ids

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

    def split(self, train_split: float, val_split: float, seed: int = 42):
        generator = torch.Generator().manual_seed(seed)

        src_count = len(self.src_ids)
        target_count = len(self.target_ids)

        src_train_count = int(train_split * src_count)
        target_train_count = int(train_split * target_count)

        src_val_count = int(val_split * src_count)
        target_val_count = int(val_split * target_count)

        src_test_count = src_count - src_train_count - src_val_count
        target_test_count = target_count - target_train_count - target_val_count

        src_train, src_valid, src_test = torch.utils.data.random_split(
            self.src_ids,
            (src_train_count, src_val_count, src_test_count),
            generator=generator,
        )

        target_train, target_valid, target_test = torch.utils.data.random_split(
            self.target_ids,
            (target_train_count, target_val_count, target_test_count),
            generator=generator,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.numpy_collate,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.numpy_collate,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lang_dataset.numpy_collate,
        )

    def get_loaders(self):
        pass
