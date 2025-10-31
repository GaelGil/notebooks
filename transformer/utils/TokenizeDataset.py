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

    def encode(self, example, tokenizer_src, tokenizer_target, max_len=128):
        src = (
            [tokenizer_src.token_to_id("[SOS]")]
            + tokenizer_src.encode(example["en"]).ids
            + [tokenizer_src.token_to_id("[EOS]")]
        )

        tgt = (
            [tokenizer_target.token_to_id("[SOS]")]
            + tokenizer_target.encode(example["de"]).ids
            + [tokenizer_target.token_to_id("[EOS]")]
        )

        src = src[:max_len] + [tokenizer_src.token_to_id("[PAD]")] * (
            max_len - len(src)
        )
        tgt = tgt[:max_len] + [tokenizer_target.token_to_id("[PAD]")] * (
            max_len - len(tgt)
        )

        return {"src_ids": src, "tgt_ids": tgt}
