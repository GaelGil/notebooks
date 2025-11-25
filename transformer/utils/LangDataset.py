import random

import jax
import jax.numpy as jnp
from datasets import load_dataset

from utils.Tokenizer import Tokenizer


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
        if dataset_name:
            self.dataset: dict = load_dataset(dataset_name)

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

    def pad_sequences(self, sequences, pad_id=0, max_len=None):
        """
        sequences: list of list of token ids
        pad_id: integer used for padding
        max_len: if None, pad to the length of the longest sequence
        """
        padded = []
        for seq in sequences:
            seq = seq[:max_len]  # truncate if too long
            padding = [pad_id] * (max_len - len(seq))
            padded.append(seq + padding)
        return jnp.array(padded, dtype=jnp.int32)

    def prep_data(self, src_data, target_data, tokenizer: Tokenizer):
        src_ids = []
        target_ids = []

        for src, target in zip(src_data, target_data):
            src_ids.append(tokenizer.encode(src, add_bos=True, add_eos=True))
            target_ids.append(tokenizer.encode(target, add_bos=True, add_eos=True))

        src_ids_padded = self.pad_sequences(
            src_ids, pad_id=tokenizer.sp.pad_id(), max_len=self.seq_len
        )
        target_ids_padded = self.pad_sequences(
            target_ids, pad_id=tokenizer.sp.pad_id(), max_len=self.seq_len
        )

        return src_ids_padded, target_ids_padded

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

        jnp.save(f"{splits_path}/train_{src_name}.npy", train_src)
        jnp.save(f"{splits_path}/val_{src_name}.npy", val_src)
        jnp.save(f"{splits_path}/test_{src_name}.npy", test_src)

        jnp.save(f"{splits_path}/train_{target_name}.npy", train_target)
        jnp.save(f"{splits_path}/val_{target_name}.npy", val_target)
        jnp.save(f"{splits_path}/test_{target_name}.npy", test_target)

        return train_src, val_src, test_src, train_target, val_target, test_target

    def load_splits(self, splits_path: str, src_name: str, target_name: str):
        train_src = jnp.load(f"{splits_path}/train_{src_name}.npy")
        val_src = jnp.load(f"{splits_path}/val_{src_name}.npy")
        test_src = jnp.load(f"{splits_path}/test_{src_name}.npy")

        train_target = jnp.load(f"{splits_path}/train_{target_name}.npy")
        val_target = jnp.load(f"{splits_path}/val_{target_name}.npy")
        test_target = jnp.load(f"{splits_path}/test_{target_name}.npy")

        return train_src, val_src, test_src, train_target, val_target, test_target
