from flax import nnx
from jax import numpy as jnp
from jax import Array


class MultiHeadAttentionBlock(nnx.Module):
    """
    Multi Head Attention Block

    In multi head attention we split the original input sequence
    (seq_len, d_model) into (n_heads, seq_len, d_k).

    Each head has its own fragments of query, key and value

    This way each head can focus on learning different representations
    of the sequence rather than learning the same representation for
    every part of the sequence.

    """

    def __init__(
        self, d_model: int, n_heads: int, dropout_rate: float, rngs: nnx.Rngs
    ) -> None:
        """

        Args:
            d_model: dimension of the model
            n_heads: number of heads
            dropout_rate: dropout probability

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
        query: Array,  # (batch_size, n_heads, seq_len, d_k)
        key: Array,  # (batch_size, n_heads, seq_len, d_k)
        value: Array,  # (batch_size, n_heads, seq_len, d_k)
        mask: Array,
        dropout: nnx.Dropout,
        is_training: bool,
        rngs: nnx.Rngs,
    ) -> Array:
        d_k = query.shape[-1]  # get dimension of last axis
        # (Q * K^T)/sqrt(d_k)
        attention_scores = jnp.matmul(query, key.swapaxes(-2, -1)) / jnp.sqrt(d_k)

        if mask is not None:
            # ensure mask shape is broadcastable to attention_scores
            if mask.ndim == 2:  # (B, Lk)
                mask = mask[:, None, None, :]  # (B, 1, 1, Lk)
            elif mask.ndim == 3:  # (B, Lq, Lk) typical for decoder self-attention
                mask = mask[:, None, :, :]  # (B, 1, Lq, Lk)
            # else assume already broadcastable (e.g., (B, H, Lq, Lk))
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
        q: Array,
        k: Array,
        v: Array,
        mask: Array,
        is_training: bool,
        rngs: nnx.Rngs,
    ):
        """

        Args:
            q: query (seq_len, d_model)
            k: key (seq_len, d_model)
            v: value (seq_len, d_model)
            mask: mask
            is_training: is training
            rngs: rngs

        Returns:
            Array
        """
        query = self.w_q(q)  # (batch_size, seq_len, d_model)
        key = self.w_k(k)  # (batch_size, seq_len, d_model)
        value = self.w_v(v)  # (batch_size, seq_len, d_model)

        # these will be the same in the encoder
        batch_size_q, seq_len_q, _ = query.shape  # decoder query
        batch_size_k, seq_len_k, d_model = key.shape  # encoder key/value

        # (batch_size, seq_len, d_model) -> (batch_size, n_heads, seq_len, d_k)
        # reshape n samples of size (seq_len, d_model) into n samples where each sample
        # has a n heads (matrices) of size (seq_len, d_k)
        query = query.reshape(
            batch_size_q, seq_len_q, self.n_heads, self.d_k
        ).transpose(0, 2, 1, 3)
        key = key.reshape(batch_size_k, seq_len_k, self.n_heads, self.d_k).transpose(
            0, 2, 1, 3
        )
        value = value.reshape(
            batch_size_k, seq_len_k, self.n_heads, self.d_k
        ).transpose(0, 2, 1, 3)

        # scaled dot product
        # (batch_size, n_heads, seq_len, d_k) -> (batch_size, n_heads, seq_len, d_k)
        x = self.scaled_dot_product_attention(
            query, key, value, mask, self.dropout, is_training, rngs=rngs
        )

        # merge heads from (batch_size, n_heads, seq_len, d_k) back to (batch_size, seq_len, D_model)
        x = x.transpose(0, 2, 1, 3).reshape(batch_size_q, seq_len_q, d_model)

        # final linear
        x = self.w_o(x)  # (batch_size, seq_len, d_model)
        return x
