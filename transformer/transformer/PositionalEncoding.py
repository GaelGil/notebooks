from flax import nnx
import jax
from jax import Array


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

        self.pe = nnx.Param(
            jax.random.normal(rngs.params(), (1, seq_len, d_model)) * 0.02
        )

    def __call__(self, x: Array, is_training: bool, rngs: nnx.Rngs):
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)
            training: whether in training mode for dropout
        """
        seq_len = x.shape[1]  # actual runtime sequence length
        pe = self.pe[:, :seq_len, :]  # slice to match

        x = x + pe
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        return self.dropout(x, deterministic=not is_training, rngs=rngs)
