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

    def write_txt(self, data: list, f: object):
        # Write list to file
        for line in data:
            # strip whitespace just in case
            line = line.strip()
            if line:  # skip empty lines
                f.write(line + "\n")

    def create_joint_corpus(self, src_one, target_one, src_two, target_two):
        # Combine into one file for tokenizer
        with open(self.config.JOINT_CORPUS_PATH, "w", encoding="utf-8") as f:
            for sentence_list in [src_one + src_two, target_one, target_two]:
                self.write_txt(data=sentence_list, f=f)

    def train_tokenizer(self, src_one, target_one, src_two, target_two):
        os.makedirs(self.config.TOKENIZER_PATH, exist_ok=True)

        self.create_joint_corpus(src_one, target_one, src_two, target_two)

        spm.SentencePieceTrainer.Train(
            input=self.config.JOINT_CORPUS_PATH,
            model_prefix=os.path.join(self.config.TOKENIZER_PATH, "joint"),
            model_type="unigram",
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )
        # Load the trained tokenizer
        self.sp.Load(self.config.TOKENIZER_MODEL_PATH)

    def load_tokenizer(self):
        self.sp.Load(self.config.TOKENIZER_MODEL_PATH)

    def encode(self, text: str, add_bos=True, add_eos=True):
        ids = self.sp.Encode(text, out_type=int)
        if add_bos:
            ids = [self.sp.bos_id()] + ids
        if add_eos:
            ids = ids + [self.sp.eos_id()]
        return ids

    def decode(self, ids):
        return self.sp.Decode(ids)
