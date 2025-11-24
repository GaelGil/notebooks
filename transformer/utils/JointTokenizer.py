import os

import sentencepiece as spm

from utils.config import Config


class JointTokenizer:
    def __init__(self, config: Config):
        self.config = config
        self.sp = spm.SentencePieceProcessor()

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()

    def train_tokenizer(self):
        os.makedirs(self.config.TOKENIZER_PATH, exist_ok=True)

        # collect all text files from config
        input_files = [
            self.config["spa_en_src"],
            self.config["spa_en_tgt"],
            self.config["spa_l2_src"],
            self.config["spa_l2_tgt"],
        ]

        merged_path = os.path.join(self.config.TOKENIZER_PATH, "all_text.txt")

        # combine text into one file
        with open(merged_path, "w", encoding="utf8") as out:
            for f in input_files:
                with open(f, "r", encoding="utf8") as inp:
                    for line in inp:
                        out.write(line.strip() + "\n")

        # Train SentencePiece Unigram (T5 style)
        self.sp = spm.SentencePieceTrainer.train(
            input=merged_path,
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
        if os.path.exists(self.config.TOKENIZER_PATH):
            self.sp.load(self.config.TOKENIZER_PATH)
            return self.sp

        else:
            return self.train_tokenizer()

    def encode(self, text: str, add_bos=True, add_eos=True):
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.sp.bos_id()] + ids
        if add_eos:
            ids = ids + [self.sp.eos_id()]
        return ids

    def decode(self, ids):
        return self.sp.decode(ids)
