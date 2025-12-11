from flax import nnx
from jax import numpy as jnp


class FeedForwardBlock(nnx.Module):
    d_model: int
    d_ff: int
    dropout_rate: float

    def setup(self) -> None:
        """
        Args:
            d_model: dimension of the model
            d_ff: dimension of the feed forward network
            dropout: dropout probability

        Returns:
            None
        """

        self.linear_1 = nnx.Linear(features=self.d_ff, dtype=jnp.float32)
        self.dropout = nnx.Linear(rate=self.dropout_rate)
        self.linear_2 = nnx.Linear(features=self.d_model, dtype=jnp.float32)

    def __call__(self, x: jnp.ndarray, is_training: bool):
        # simple feed forward network
        # (seq_len, d_model) --> (dff, d_model) --> (seq_len, d_model)
        x = nnx.leaky_relu(self.linear_1(x))
        x = self.dropout(x, deterministic=not is_training)
        x = self.linear_2(x)
        return x
