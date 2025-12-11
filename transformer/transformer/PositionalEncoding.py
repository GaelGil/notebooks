from flax import nnx
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

        self.dropout = nnx.Dropout(rate=dropout_rate)

        self.pe = CustomVariable(
            jnp.zeros((1, seq_len, d_model), dtype=jnp.float32), mutable=False
        )

    def __call__(self, x: jnp.ndarray, is_training: bool):
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)
            training: whether in training mode for dropout
        """
        seq_len = x.shape[1]  # actual runtime sequence length
        pe = self.pe[:, :seq_len, :]  # slice to match

        x = x + pe
        return self.dropout(x, deterministic=not is_training)
