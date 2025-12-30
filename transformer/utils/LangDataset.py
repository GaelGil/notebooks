import random

import numpy as np
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
        splits_path: str = None,
    ):
        """
        Load the dataset from the Hugging Face Hub
        Args:
            src_lang: source language
            target_lang: target language
            dataset_name: dataset name
            src_file: path to the source file
            target_file: path to the target file
            seq_len: sequence length

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
        self.paths = {str: str}
        self.splits_path: str = splits_path

    def load_data(self):
        """
        Load the dataset from the Hugging Face Hub or from a file
        Args:
            None

        Returns:
            src_data: source data
            target_data: target data
        """
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

    def valid_pair(self, example: dict):
        """
        Check if the example is a valid pair of source and target

        Args:
            example: example to check

        Returns:
            True if the example is a valid pair, False otherwise
        """
        # Get source and target
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

    def buil_save_paths(self):
        """
        Build and save paths for the splits
        Args:
            splits_path: path to save the splits

        Returns:
            None
        """
        train_path = f"{self.splits_path}/train"
        val_path = f"{self.splits_path}/val"
        test_path = f"{self.splits_path}/test"

        train_src_path = f"{train_path}/_{self.src_lang}.npy"
        val_src_path = f"{val_path}/_{self.src_lang}.npy"
        test_src_path = f"{test_path}/_{self.src_lang}.npy"

        train_target_path = f"{train_path}/_{self.target_lang}.npy"
        val_target_path = f"{val_path}/_{self.target_lang}.npy"
        test_target_path = f"{test_path}/_{self.target_lang}.npy"

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        self.paths = {
            "train_src": train_src_path,
            "val_src": val_src_path,
            "test_src": test_src_path,
            "train_target": train_target_path,
            "val_target": val_target_path,
            "test_target": test_target_path,
        }

    def split(
        self,
        src: list,
        target: list,
        train_size: float,
        val_size: float,
    ):
        """
        Split the dataset into train, val and test sets

        Args:
            src: source data
            target: target data
            train_size: size of the train set
            val_size: size of the validation set
            src_name: name of the source language
            target_name: name of the target language

        Returns:
            None
        """
        os.makedirs(self.splits_path, exist_ok=True)
        rng = np.random.default_rng(seed=0)
        indices = rng.permutation(len(src))
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

        self.buil_save_paths()

        np.save(self.paths["train_src"], train_src)
        np.save(self.paths["val_src"], val_src)
        np.save(self.paths["test_src"], test_src)

        np.save(self.paths["train_target"], train_target)
        np.save(self.paths["val_target"], val_target)
        np.save(self.paths["test_target"], test_target)
