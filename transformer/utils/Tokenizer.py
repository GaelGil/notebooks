import os
import jax.numpy as jnp
import sentencepiece as spm


class Tokenizer:
    def __init__(
        self,
        corpus_path: str = None,
        tokenizer_path: str = None,
        tokenizer_model_path: str = None,
        model_prefix: str = None,
        seq_len: int = None,
        prefix: str = None,
    ):
        self.corpus_path = corpus_path
        self.tokenizer_path = tokenizer_path
        self.tokenizer_model_path = tokenizer_model_path
        self.sp = spm.SentencePieceProcessor()
        self.vocab_size = None
        self.model_prefix = model_prefix
        self.seq_len = seq_len
        self.prefix: str = prefix

    def write_txt(self, data: list, f: object):
        # Write list to file
        for line in data:
            # strip whitespace just in case
            line = line.strip()
            if line:  # skip empty lines
                f.write(line + "\n")

    def create_joint_corpus(self, src, target, target_one):
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
            for sentence_list in [src, target, target_one]:
                self.write_txt(data=sentence_list, f=f)

    def train_tokenizer(
        self,
        src: list[str],
        target: list[str],
        src_one: list[str] = [],
        target_one: list[str] = [],
        prefixs=None,
    ):
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
        self.create_joint_corpus(src + src_one, target, target_one)
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

    def pad_sequences(
        self, sequences: list, pad_id: int = 0, max_len: int = None
    ) -> jnp.ndarray:
        """
        Args:
            sequences: list of list of token ids
            pad_id: integer used for padding
            max_len: if None, pad to the length of the longest sequence

        Returns:
            padded: numpy array of shape [N, max_len]
        """
        padded = []
        for seq in sequences:
            seq = seq[:max_len]  # truncate if too long
            padding = [pad_id] * (max_len - len(seq))
            padded.append(seq + padding)
        return jnp.array(padded, dtype=jnp.int32)

    def prep_data(self, src, target, prefix=None):
        src_ids = []
        target_ids = []

        # prefix = f"Translate {src_fname} to {target_fname} "
        # prefix = tokenizer.encode(text=prefix, add_bos=False, add_eos=False)
        for src, target in zip(src, target):
            # encode and add bos and eos
            src_ids.append(
                self.encode(
                    text=src,
                    add_bos=False,
                    add_eos=False,
                    prefix=prefix,
                )
            )
            target_ids.append(
                self.encode(
                    text=target,
                    add_bos=True,
                    add_eos=True,
                )
            )
        src_seq_len = 0
        target_seq_len = 0
        max_src_seq_len = 0
        max_target_seq_len = 0
        for i in range(len(src_ids)):
            src_seq_len += len(src_ids[i])
            target_seq_len += len(target_ids[i])
            if len(src_ids[i]) > max_src_seq_len:
                max_src_seq_len = len(src_ids[i])
            if len(target_ids[i]) > max_target_seq_len:
                max_target_seq_len = len(target_ids[i])

        # pad sequences up to seq_len
        src_ids_padded: jnp.ndarray = self.pad_sequences(
            src_ids, pad_id=self.sp.pad_id(), max_len=self.seq_len
        )
        target_two_ids_padded: jnp.ndarray = self.pad_sequences(
            target_ids, pad_id=self.sp.pad_id(), max_len=self.seq_len
        )

        return src_ids_padded, target_two_ids_padded

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
