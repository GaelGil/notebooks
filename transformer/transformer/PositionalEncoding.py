from flax import nnx
from jax import numpy as jnp


class PositionalEncoding(nnx.Module):
    d_model: int
    seq_len: int
    dropout_rate: float

    def setup(self):
        """
        Args:
            d_model: embedding dimension
            seq_len: maximum input sequence length
            dropout: dropout rate (used during training)
        """

        self.dropout = nnx.Dropout(rate=self.dropout_rate)

        self.pe = nnx.Param(
            "positional_encoding",
            nnx.initializers.normal(stddev=0.02),
            (1, self.seq_len, self.d_model),
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
