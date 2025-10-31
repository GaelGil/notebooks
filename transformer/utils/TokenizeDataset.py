from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path


class TokenizeDataset:
    def __init__(
        self,
        dataset,
        language,
        tokenizer_path,
    ):
        self.language = language
        self.tokenizer_path = Path(tokenizer_path.format(language))
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        text_iter = (example[language] for example in dataset)
        tokenizer.train_from_iterator(text_iter, trainer=trainer)
        tokenizer.save(str(self.tokenizer_path))
        self.tokenizer = tokenizer

    def get_tokenizer(self):
        return self.tokenizer

    def encode(self, text):
        return self.tokenizer.encode(text)
