from flax import nnx
from jax import numpy as jnp


class MultiHeadAttentionBlock(nnx.Module):
    """
    Multi Head Attention Block
    """

    d_model: int
    n_heads: int
    dropout_rate: float

    def setup(self) -> None:
        """

        Args:
            d_model: dimension of the model
            n_heads: number of heads
            dropout: dropout probability

        Returns:
            None
        """

        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

        self.attention = nnx.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
        )

    def __call__(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        mask: jnp.ndarray,
        is_training: bool,
    ) -> jnp.ndarray:
        """

        Args:
            q: query
            k: key
            v: value

        Returns:
            jnp.ndarray
        """

        return self.attention(
            inputs_q=q,
            inputs_k=k,
            inputs_v=v,
            mask=mask,
            deterministic=not is_training,
        )
