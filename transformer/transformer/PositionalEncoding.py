from flax import nnx
from jax import Array
from jax import numpy as jnp

class CustomVariable(nnx.Variable):
    pass


class PositionalEncoding(nnx.Module):
    def __init__(self, d_model: int, seq_len: int, dropout_rate: float, rngs: nnx.Rngs):
        """
        Args:
            d_model: embedding dimension
            seq_len: maximum input sequence length
            dropout: dropout rate (used during training)
        """

        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

        self.pe = nnx.Embed(
            num_embeddings=seq_len,
            features=d_model,
            embedding_init=nnx.initializers.normal(stddev=0.02),
            rngs=rngs,
        )

    def __call__(self, x: Array, is_training: bool, rngs: nnx.Rngs):
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)
            training: whether in training mode for dropout
        """
        B, T, _ = x.shape
        positions = jnp.arange(T)[None, :]  # (1, T)
        x = x + self.pe(positions)     # broadcasts to (B,T,D)
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        return self.dropout(x, deterministic=not is_training, rngs=rngs)
