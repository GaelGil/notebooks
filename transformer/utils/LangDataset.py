import jax.numpy as jnp
import torch
from datasets import load_dataset
from jax.tree_util import tree_map
from torch.utils.data import default_collate


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
