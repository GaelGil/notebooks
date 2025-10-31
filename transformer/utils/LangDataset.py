from datasets import load_dataset
from torch.utils.data import DataLoader
# Load a public dataset
# Replace "username/my_dataset" with the actual dataset ID from the Hugging Face Hub


class LangDataset:
    def __init__(self, dataset_name: str):
        self.dataset = None
        self.dataset_name = dataset_name
        pass

    def load_dataset(self):
        self.dataset = load_dataset(self.dataset_name)

        return self.dataset

    def length(self):
        return len(self.dataset["train"])

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
