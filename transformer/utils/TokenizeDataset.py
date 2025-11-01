from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer


class TokenizeDataset:
    def __init__(
        self,
        dataset,
        language,
        tokenizer_path,
    ):
        """
        Tokenize the dataset

        Args:
            dataset: dataset to tokenize
            language: language to tokenize
            tokenizer_path: path to save the tokenizer

        Returns:
            None
        """
        self.language = language
        self.dataset = dataset
        self.tokenizer_path = Path(tokenizer_path.format(language))
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )

        def text_iter():
            for example in dataset:
                text = example.get(language)
                if text:
                    yield text

        tokenizer.train_from_iterator(text_iter(), trainer=trainer)
        tokenizer.save(str(self.tokenizer_path))
        self.tokenizer = tokenizer

    def get_tokenizer(self) -> Tokenizer:
        """
        Get the tokenizer

        Args:
            None

        Returns:
            Tokenizer
        """
        return self.tokenizer

    def check_null(self):
        """
        Check if the tokenizer is null

        Args:
            None

        Returns:
            True if the tokenizer is null
        """
        for example in self.dataset:
            text = example.get(self.language)
            if not isinstance(text, str) or not text.strip():
                self.dataset.remove(example)
        return True

    def get_token_ids(self):
        """
        Get the token ids for the dataset

        Args:
            None

        Returns:
            List of token ids
        """
        token_ids = []
        # for example in self.dataset:
        for example in self.dataset:
            # get the text with the corresponding language
            text = example.get(self.language)
            token_ids.append(self.encode(text))
        return token_ids

    def encode(self, text, max_len=128) -> dict:
        """
        Encode the text


        Args:
            text: example from the dataset
            max_len: maximum length of the sequence

        Returns:
            Dict containing the encoded sequence

        """
        token_ids = (
            [self.tokenizer.token_to_id("[SOS]")]
            + self.tokenizer.encode(text).ids
            + [self.tokenizer.token_to_id("[EOS]")]
        )

        token_ids = token_ids[:max_len] + [self.tokenizer.token_to_id("[PAD]")] * (
            max_len - len(token_ids)
        )

        return {
            "ids": token_ids,
        }
