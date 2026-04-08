from flax import nnx
from jax import Array
from jax import numpy as jnp


class MultiHeadLatentAttention(nnx.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_latent: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ) -> None:
        """

        Args:
            d_model: dimension of the model
            n_heads: number of heads
            dropout_rate: dropout probability
            rngs: rngs

        Returns:
            None
        """

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.d_latent = d_latent
        self.w_q = nnx.Linear(in_features=d_model, out_features=d_model, rngs=rngs)

        self.w_kv_down = nnx.Linear(
            in_features=d_model, out_features=d_latent, rngs=rngs
        )
        self.w_k_up = nnx.Linear(in_features=d_latent, out_features=d_model, rngs=rngs)
        self.w_v_up = nnx.Linear(in_features=d_latent, out_features=d_model, rngs=rngs)

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
            if mask.ndim == 2:
                mask = mask[:, None, None, :]  # (B, 1, 1, Lk)
            elif mask.ndim == 3:  # (B, Lq, Lk) typical for decoder self-attention
                mask = mask[:, None, :, :]  # (B, 1, Lq, Lk)

            neg_inf = jnp.finfo(attention_scores.dtype).min
            # else assume already broadcastable (e.g., (B, H, Lq, Lk))
            attention_scores = jnp.where(mask, attention_scores, neg_inf)
        # softmax(Q * K^T/sqrt(d_k))
        attention_scores = nnx.softmax(
            attention_scores.astype(jnp.float32), axis=-1
        ).astype(query.dtype)

        attention_scores = dropout(
            attention_scores, deterministic=not is_training, rngs=rngs
        )
        # (Q * K^T)/sqrt(d_k) * V
        x = jnp.matmul(attention_scores, value)

        return x

    @staticmethod
    def split_heads(self, x: Array) -> Array:
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(
            0, 2, 1, 3
        )

    @staticmethod
    def combine_heads(self, x: Array) -> Array:
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

    def __call__(
        self,
        x: Array,
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

        q = self.w_q(
            x
        )  # (batch_size, seq_len, d_model) --> (batch_size, d_model, d_model)

        kv_latent = self.w_kv_down(
            x
        )  # (batch_size, seq_len, d_model) --> (batch_size, d_model, d_latent)

        k = self.w_k_up(
            kv_latent
        )  # (batch_size, d_model, d_latent) --> (batch_size, d_model, d_model)
        v = self.w_v_up(
            kv_latent
        )  # (batch_size, d_model, d_latent) --> (batch_size, d_model, d_model)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        x = self.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            mask=mask,
            dropout=self.dropout,
            is_training=is_training,
            rngs=rngs,
        )

        x = self.combine_heads(x)

        x = self.w_o(x)
        return x
