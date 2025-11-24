import os
import random

from datasets import load_dataset


class LangDataset:
    def __init__(
        self,
        src_lang: str,
        target_lang: str,
        dataset_name: str = None,
        src_file: str = None,
        target_file: str = None,
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
        self.src_data = None
        self.target_data = None
        if dataset_name:
            self.dataset: dict = load_dataset(dataset_name)

    def load_data(self):
        if self.src_file:
            assert os.path.exists(self.src_file)
            assert os.path.exists(self.target_file)

            with open(self.src_file, "r", encoding="utf-8") as f:
                src = [line.strip() for line in f]

            with open(self.target_file, "r", encoding="utf-8") as f:
                target = [line.strip() for line in f]

            assert len(src) == len(target), "src and tgt sizes mismatch"

            combined = list(zip(src, target))
            random.shuffle(combined)
            src, target = zip(*combined)

            self.src_data = list(src)
            self.target_data = list(target)
            return self.src_data, self.target_data
        else:
            self.src_data = self.dataset["train"][self.src_lang]
            self.target_data = self.dataset["train"][self.target_lang]

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
