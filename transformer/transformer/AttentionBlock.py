from flax import nnx
from jax import numpy as jnp


class MultiHeadAttentionBlock(nnx.Module):
    """
    Multi Head Attention Block
    """

    def __init__(
        self, d_model: int, n_heads: int, dropout_rate: float, rngs: nnx.Rngs
    ) -> None:
        """

        Args:
            d_model: dimension of the model
            n_heads: number of heads
            dropout: dropout probability

        Returns:
            None
        """

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.attention = nnx.MultiHeadAttention(
            num_heads=n_heads,
            in_features=d_model,
            qkv_features=d_model,
            dropout_rate=dropout_rate,
            rngs=rngs,
            decode=False,
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
