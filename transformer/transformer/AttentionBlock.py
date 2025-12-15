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
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.w_q = nnx.Linear(in_features=d_model, out_features=d_model, rngs=rngs)
        self.w_k = nnx.Linear(in_features=d_model, out_features=d_model, rngs=rngs)
        self.w_v = nnx.Linear(in_features=d_model, out_features=d_model, rngs=rngs)
        self.w_o = nnx.Linear(in_features=d_model, out_features=d_model, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate)

    @staticmethod
    def scaled_dot_product_attention(
        query: jnp.ndarray,  # (B, H, Q, Dk)
        key: jnp.ndarray,  # (B, H, K, Dk)
        value: jnp.ndarray,  # (B, H, K, Dk)
        mask: jnp.ndarray,
        dropout: nnx.Dropout,
        is_training: bool,
        rngs: nnx.Rngs,
    ) -> jnp.ndarray:
        d_k = query.shape[-1]  # get dimension of last axis
        # (Q * K^T)/sqrt(d_k)
        attention_scores = jnp.matmul(query, key.swapaxes(-2, -1)) / jnp.sqrt(d_k)

        if mask is not None:
            # ensure mask shape is broadcastable to attention_scores
            if mask.ndim == 2:  # (B, Lk)
                mask = mask[:, None, None, :]  # (B, 1, 1, Lk)
            attention_scores = jnp.where(mask == 0, -1e10, attention_scores)

        # softmax(Q * K^T/sqrt(d_k))
        attention_scores = nnx.softmax(attention_scores, axis=-1)
        if dropout:
            attention_scores = dropout(
                attention_scores, deterministic=not is_training, rngs=rngs
            )
        # (Q * K^T)/sqrt(d_k) * V
        x = jnp.matmul(attention_scores, value)

        return x

    def __call__(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        mask: jnp.ndarray,
        is_training: bool,
        rngs: nnx.Rngs,
    ):
        """

        Args:
            q: query
            k: key
            v: value
            mask: mask

        Returns:
            None
        """
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        Bq, Lq, _ = query.shape  # decoder query
        Bk, Lk, d_model = key.shape  # encoder key/value

        query = query.reshape(Bq, Lq, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        key = key.reshape(Bk, Lk, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        value = value.reshape(Bk, Lk, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # attention
        x = self.scaled_dot_product_attention(
            query, key, value, mask, self.dropout, is_training, rngs=rngs
        )

        # merge heads -> (B, L, D_model)
        x = x.transpose(0, 2, 1, 3).reshape(Bq, Lq, d_model)

        # final linear
        x = self.w_o(x)
        return x
