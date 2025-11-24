import os

import sentencepiece as spm

from utils.config import Config


class Tokenizer:
    def __init__(self, config: Config):
        self.config = config
        self.sp = spm.SentencePieceProcessor()

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()

    def train_tokenizer(self, src_one, target_one, src_two, target_two):
        os.makedirs(self.config.TOKENIZER_PATH, exist_ok=True)

        # combine text into one file
        temp_file = self.config.TOKENIZER_PATH + "_train.txt"
        with open(temp_file, "w", encoding="utf-8") as f:
            for s, t in zip(self.src_texts, self.tgt_texts):
                f.write(s + "\n")
                f.write(t + "\n")

        # Train SentencePiece Unigram (T5 style)
        self.sp = spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=os.path.join(self.config.TOKENIZER_PATH, "joint"),
            vocab_size=self.config["vocab_size"],
            model_type="unigram",
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )

        return self.sp

    def load_tokenizer(self):
        self.sp.load(self.config.TOKENIZER_MODEL_PATH)
        return self.sp

    def encode(self, text: str, add_bos=True, add_eos=True):
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.sp.bos_id()] + ids
        if add_eos:
            ids = ids + [self.sp.eos_id()]
        return ids

    def decode(self, ids):
        return self.sp.decode(ids)
