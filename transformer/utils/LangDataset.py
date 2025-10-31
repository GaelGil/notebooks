from datasets import load_dataset
from torch.utils.data import DataLoader
# Load a public dataset
# Replace "username/my_dataset" with the actual dataset ID from the Hugging Face Hub


class LangDataset:
    def __init__(self):
        self.dataset = None
        pass

    def load_dataset(self):
        self.dataset = load_dataset("somosnlp-hackathon-2022/Axolotl-Spanish-Nahuatl")

        return self.dataset

    def length(self):
        return len(self.dataset)

    def split(self, test_size=0.1, val_size=0.1, seed=42):
        # Split training into train/val if needed
        train_valid = self.dataset["train"].train_test_split(
            test_size=val_size, seed=seed
        )
        train_ds = train_valid["train"]
        val_ds = train_valid["test"]

        # If dataset already has a test split, use it
        test_ds = self.dataset.get("test", None)

        return train_ds, val_ds, test_ds

    def to_data_loader(self, train_ds, val_ds, batch_size=32):
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)

        return train_loader, val_loader
