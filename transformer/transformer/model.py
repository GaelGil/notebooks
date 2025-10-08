from flax import nnx


class InputEmbeddings(nnx.Module):
    def __init__(self, d_model, int, vocab_size: int) -> None:
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nnx.Embedding(vocab_size, d_model)

    def __call__(self, x):
        return self.embedding(x) * self.d_model**0.5


class PositionalEncoding(nnx.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        self.d_model = d_model
        self.seq_len = seq_len
