from datasets import load_dataset


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

    def load_data(self):
        self.src_data = self.dataset["train"][self.src_lang]
        self.target_data = self.dataset["train"][self.target_lang]
