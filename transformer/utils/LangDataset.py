from datasets import load_dataset
from torch.utils.data import DataLoader

# Load a public dataset
# Replace "username/my_dataset" with the actual dataset ID from the Hugging Face Hub


class LangDataset:
    def __init__(self, dataset_name: str, src_lang: str, target_lang: str):
        self.dataset = None
        self.dataset_name = dataset_name
        self.src_lang = src_lang
        self.target_lang = target_lang
        pass

    def load_dataset(self):
        """
        Load the dataset from the Hugging Face Hub
        Args:
            None

        Returns:
            None
        """
        self.dataset = load_dataset(self.dataset_name)
        return self.dataset

    def length(self):
        """
        Get the length of the dataset
        Args:
            None

        Returns:
            None
        """
        return len(self.dataset["train"])

    def valid_pair(self, example):
        if not self.dataset:
            print("Dataset not loaded")
            return
        return (
            isinstance(example[self.src_lang], str)
            and example[self.src_lang].strip()
            and isinstance(example[self.target_lang], str)
            and example[self.target_lang].strip()
        )

    def handle_null(self):
        """
        Check if the tokenizer is null

        Args:
            None

        Returns:
            None
        """
        self.dataset = self.dataset.filter(self.valid_pair)

    def split(self, test_split: float, val_split: float, seed: int = 42):
        train_valid = self.dataset["train"].train_test_split(
            test_size=(val_split + test_split), seed=seed
        )
        train_ds = train_valid["train"]
        val_ds = train_valid["test"]

        test_ds = val_ds.train_test_split(test_size=test_split, seed=seed)["test"]
        return train_ds, val_ds, test_ds

    def to_data_loader(self, train_ds, val_ds, batch_size=32):
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)

        return train_loader, val_loader
