import os

import sentencepiece as spm


class Tokenizer:
    def __init__(
        self,
        corpus_path: str = None,
        tokenizer_path: str = None,
        tokenizer_model_path: str = None,
        model_prefix: str = None,
    ):
        self.corpus_path = corpus_path
        self.tokenizer_path = tokenizer_path
        self.tokenizer_model_path = tokenizer_model_path
        self.sp = spm.SentencePieceProcessor()
        self.vocab_size = None
        self.model_prefix = model_prefix

    def write_txt(self, data: list, f: object):
        # Write list to file
        for line in data:
            # strip whitespace just in case
            line = line.strip()
            if line:  # skip empty lines
                f.write(line + "\n")

    def create_joint_corpus(self, text_one: list[str], text_two: list[str]):
        """
        Creates a joint for the text

        Args:
            text_one: list of strings
            text_two: list of strings

        Returns:
            None
        """
        os.makedirs(os.path.dirname(self.corpus_path), exist_ok=True)
        # Combine into one file for tokenizer
        with open(self.corpus_path, "w", encoding="utf-8") as f:
            for sentence_list in [text_one, text_two]:
                print(sentence_list)
                print()
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
        # Create joint corpus
        self.create_joint_corpus(text_one, text_two)
        os.makedirs(self.tokenizer_model_path, exist_ok=True)
        # Train tokenizer
        spm.SentencePieceTrainer.Train(
            input=self.corpus_path,
            model_prefix=os.path.join(self.tokenizer_model_path, self.model_prefix),
            model_type="unigram",
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=prefixs,
        )
        self.load_tokenizer()

    def get_vocab_size(self):
        return self.sp.GetPieceSize()

    def load_tokenizer(self):
        self.sp.Load(f"{self.tokenizer_model_path}/{self.model_prefix}.model")

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
