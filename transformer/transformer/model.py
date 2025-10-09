import jax.numpy as jnp
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
        self.dropout = dropout
        # TODO: updated positional encoding
        self.pe = jnp.zeros((seq_len, d_model))
        self.pos = jnp.arange(0, seq_len)[:, jnp.newaxis]
        self.scale = jnp.ones((seq_len, 1))

    def __call__(self, x):
        return x


class LayerNorm(nnx.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    # TODO: updated layer norm with biases
    def __call__(self, x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        std = jnp.std(x, axis=-1, keepdims=True)
        return x - mean / (std + self.eps)


class FeedForwardBlock(nnx.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.linear_1 = nnx.Linear(d_model, d_ff)
        self.dropout_1 = nnx.Dropout(dropout)
        self.linear_2 = nnx.Linear(d_ff, d_model)

    def __call__(self, x):
        x = nnx.leaky_relu(self.linear_1(x))
        x = self.dropout_1(x)
        x = self.linear_2(x)
        return x


class MultiHeadAttentionBlock(nnx.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.dk = d_model // n_heads
        self.w_q = nnx.Linear(d_model, d_model)
        self.w_k = nnx.Linear(d_model, d_model)
        self.w_v = nnx.Linear(d_model, d_model)
        self.w_o = nnx.Linear(d_model, d_model)
        self.dropout = nnx.Dropout(dropout)

    def __call__(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        query = query.view(
            query.shape[0], query.shape[1], self.n_heads, self.dk
        ).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.dk).transpose(
            1, 2
        )
        value = value.view(
            value.shape[0], value.shape[1], self.n_heads, self.dk
        ).transpose(1, 2)
