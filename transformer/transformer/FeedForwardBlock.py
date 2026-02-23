from flax import nnx
from jax import numpy as jnp
from jax import Array


class FeedForwardBlock(nnx.Module):
    """
    Feed forward block
    Attributes:
        linear_1: nnx.Linear
        dropout: nnx.Dropout
        linear_2: nnx.Linear
    """

    def __init__(
        self, d_model: int, d_ff: int, dropout_rate: float, rngs: nnx.Rngs
    ) -> None:
        """
        Args:
            d_model: dimension of the model
            d_ff: dimension of the feed forward network
            dropout: dropout probability

        Returns:
            None
        """

        self.linear_1 = nnx.Linear(
            in_features=d_model, out_features=d_ff, dtype=jnp.float32, rngs=rngs
        )
        self.dropout = nnx.Dropout(rate=dropout_rate)
        self.linear_2 = nnx.Linear(
            in_features=d_ff, out_features=d_model, dtype=jnp.float32, rngs=rngs
        )

    def __call__(self, x: Array, is_training: bool, rngs: nnx.Rngs):
        # simple feed forward network
        # (seq_len, d_model) --> (dff, d_model) --> (seq_len, d_model)
        x = nnx.gelu(self.linear_1(x))
        x = self.dropout(x, deterministic=not is_training, rngs=rngs)
        x = self.linear_2(x)
        return x
