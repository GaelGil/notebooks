import os

import sentencepiece as spm


class Tokenizer:
    def __init__(
        self,
        joint_corpus_path: str = None,
        tokenizer_model_path: str = None,
        tokenizer_path: str = None,
    ):
        self.joint_corpus_path = joint_corpus_path
        self.tokenizer_model_path = tokenizer_model_path
        self.tokenizer_path = tokenizer_path
        self.sp = spm.SentencePieceProcessor()
        self.vocab_size = None

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

    def create_joint_corpus(self, text_one, text_two):
        """
        Creates a joint corpus of source and target data

        Args:
            src_one: list of strings
            target_one: list of strings
            src_two: list of strings
            target_two: list of strings

        Returns:
            None
        """
        # Combine into one file for tokenizer
        with open(self.joint_corpus_path, "w", encoding="utf-8") as f:
            for sentence_list in [text_one, text_two]:
                self.write_txt(data=sentence_list, f=f)

    def train_tokenizer(self, text_one, text_two, prefixs=None):
        """
        Trains a sentencepiece tokenizer on the joint corpus

        Args:
            src_one: list of strings
            target_one: list of strings
            src_two: list of strings
            target_two: list of strings

        Returns:
            None
        """
        os.makedirs(self.tokenizer_path, exist_ok=True)
        # Create joint corpus
        self.create_joint_corpus(text_one, text_two)
        # Train tokenizer
        spm.SentencePieceTrainer.Train(
            input=self.joint_corpus_path,
            model_prefix=os.path.join(self.tokenizer_path, "joint"),
            model_type="unigram",
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=prefixs,
        )

        self.vocab_size = self.sp.get_piece_size()

        # Load the trained tokenizer
        self.load_tokenizer()

    def load_tokenizer(self):
        self.sp.Load(self.tokenizer_model_path)

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        prefix: str = None,
    ):
        """
        Args:
            text: string
            add_bos: boolean
            add_eos: boolean

        Returns:
            ids: list of integers
        """
        ids = self.sp.Encode(text, out_type=int)
        if prefix:
            ids = self.sp.Encode(prefix, out_type=int) + ids
        if add_bos:
            ids = [self.sp.bos_id()] + ids
        if add_eos:
            ids = ids + [self.sp.eos_id()]
        return ids

    def decode(self, ids):
        return self.sp.Decode(ids)
