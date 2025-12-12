import random

import jax
import jax.numpy as jnp
from datasets import load_dataset
import os


class LangDataset:
    def __init__(
        self,
        src_lang: str,
        target_lang: str,
        dataset_name: str = None,
        src_file: str = None,
        target_file: str = None,
        seq_len: int = None,
    ):
        """
        Load the dataset from the Hugging Face Hub
        Args:
            dataset_name: name of the dataset
            src_lang: source language
            target_lang: target language

        Returns:
            None
        """
        self.src_lang: str = src_lang
        self.target_lang: str = target_lang
        self.src_file = src_file
        self.target_file = target_file
        self.seq_len: int = seq_len
        self.dataset: dict = {}
        self.dataset_name: str = dataset_name
        self.paths = None

    def load_data(self):
        if self.src_file:
            with open(self.src_file, "r", encoding="utf-8") as f:
                src = [line.strip() for line in f]

            with open(self.target_file, "r", encoding="utf-8") as f:
                target = [line.strip() for line in f]

            assert len(src) == len(target), "src and tgt sizes mismatch"

            combined = list(zip(src, target))
            random.shuffle(combined)
            src, target = zip(*combined)

            src_data = list(src)
            target_data = list(target)
            return src_data, target_data
        else:
            self.dataset: dict = load_dataset(self.dataset_name)
            self.handle_null()
            src_data = self.dataset["train"][self.src_lang]
            target_data = self.dataset["train"][self.target_lang]
            return list(src_data), (target_data)

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

    def split(
        self,
        src: list,
        target: list,
        train_size: float,
        val_size: float,
        src_name: str,
        target_name: str,
        splits_path: str,
    ):
        os.makedirs(splits_path, exist_ok=True)
        indices = jnp.arange(len(src))
        indices = jax.random.permutation(key=jax.random.PRNGKey(0), x=indices)
        src = src[indices]
        target = target[indices]

        n_total = len(src)
        n_train = int(train_size * n_total)
        n_val = int(val_size * n_total)

        train_src, train_target = src[:n_train], target[:n_train]
        val_src, val_target = (
            src[n_train : n_train + n_val],
            target[n_train : n_train + n_val],
        )
        test_src, test_target = src[n_train + n_val :], target[n_train + n_val :]

        jnp.save(f"{splits_path}/train/_{src_name}.npy", train_src)
        jnp.save(f"{splits_path}/val/_{src_name}.npy", val_src)
        jnp.save(f"{splits_path}/test/_{src_name}.npy", test_src)

        jnp.save(f"{splits_path}/train/_{target_name}.npy", train_target)
        jnp.save(f"{splits_path}/val/_{target_name}.npy", val_target)
        jnp.save(f"{splits_path}/test/_{target_name}.npy", test_target)

        self.paths = {
            "train_src": f"{splits_path}/train/_{src_name}.npy",
            "val_src": f"{splits_path}/val/_{src_name}.npy",
            "test_src": f"{splits_path}/test/_{src_name}.npy",
            "train_target": f"{splits_path}/train/_{target_name}.npy",
            "val_target": f"{splits_path}/val/_{target_name}.npy",
            "test_target": f"{splits_path}/test/_{target_name}.npy",
        }

    def load_splits(self, splits_path: str, src_name: str, target_name: str):
        train_src = jnp.load(f"{splits_path}/train/_{src_name}.npy")
        val_src = jnp.load(f"{splits_path}/val/_{src_name}.npy")
        test_src = jnp.load(f"{splits_path}/test/_{src_name}.npy")

        train_target = jnp.load(f"{splits_path}/train/_{target_name}.npy")
        val_target = jnp.load(f"{splits_path}/val/_{target_name}.npy")
        test_target = jnp.load(f"{splits_path}/test/_{target_name}.npy")

        return train_src, val_src, test_src, train_target, val_target, test_target
