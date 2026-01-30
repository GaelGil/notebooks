from flax import nnx
from jax import Array
from jax import numpy as jnp


class MultiHeadAttentionBlock(nnx.Module):
    """
    Multi Head Attention Block

    Attributes:
        d_model: dimension of the model
        n_heads: number of heads
        dropout_rate: dropout rate
    """

    def __init__(
        self, d_model: int, n_heads: int, dropout_rate: float, rngs: nnx.Rngs
    ) -> None:
        """
        Set up multi head attention block. Each query, key, value gets its own linear layer. We will multiply
        w_q by q -> (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model) then split into
        n_heads -> (batch_size, seq_len, n_heads, d_k). Which means we have a sequence length each with 8 heads of size d_k
        We will then compute scaled dot product attention for each head.

        Args:
            None

        Returns:
            None
        """

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.w_q = nnx.Linear(in_features=d_model, out_features=d_model, rngs=rngs)
        self.w_k = nnx.Linear(in_features=d_model, out_features=d_model, rngs=rngs)
        self.w_v = nnx.Linear(in_features=d_model, out_features=d_model, rngs=rngs)
        self.w_o = nnx.Linear(in_features=d_model, out_features=d_model, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.n_heads = n_heads
        self.d_model = d_model

    @staticmethod
    def scaled_dot_product_attention(
        query: Array,
        key: Array,
        value: Array,
        dropout: nnx.Dropout,
        d_k: int,
        training: bool,
    ) -> Array:
        """
        For each head, compute  softmax(Q * K^T/sqrt(d_k)) * V
        Args:
            query: query
            key: key
            value: value
            dropout: dropout

        Returns:
            None
        """

        # (Q * K^T)/sqrt(d_k)
        attention_scores = jnp.matmul(query, key.swapaxes(-2, -1)) / jnp.sqrt(d_k)
        # softmax(Q * K^T/sqrt(d_k))
        attention_scores = nnx.softmax(attention_scores, axis=-1)
        if dropout:
            attention_scores = dropout(attention_scores, deterministic=not training)
        # softmax((Q * K^T)/sqrt(d_k)) * V
        x = jnp.matmul(attention_scores, value)
        return x

    def __call__(
        self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, is_training: bool
    ):
        """

        Args:
            q: query
            k: key
            v: value

        Returns:
            None
        """
        # (seq_len, d_model) * (d_model, d_model) -> (seq_len, d_model)
        query: jnp.ndarray = self.w_q(q)
        key: jnp.ndarray = self.w_k(k)
        value: jnp.ndarray = self.w_v(v)

        # (seq_len, d_model) -> (seq_len, n_heads, d_k) -> (n_heads, seq_len, d_k)
        # where dk = d_model // n_heads
        # (n, 512) -> (n, h, dk) -> (h, n, dk)
        # (3, 512) -> (3, 8, 64) -> (8, 3, 64)
        #
        # Sequence length n where each token is of dimension 512 ->
        # Sequence length n where each token is an array of 8 vectors of dimension 64 ->
        # Explaination: In a sequence the embeddings are split into 8 parts so that each head can focus on different parts of the embeddings
        # 8 Heads where each head contains a matrix of n vectors of dimension 64
        # keep the batch dimension the same and the sequence length the same
        # split the embeddings into 8 heads

        # (batch_size, seq_len, d_model), since we are only using encoder
        # we only use the query
        B, L, _, _ = query.shape
        query = query.reshape(B, L, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        key = key.reshape(B, L, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        value = value.reshape(B, L, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # apply scaled dot product attention to each head
        x = MultiHeadAttentionBlock.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            dropout=self.dropout,
            d_k=self.d_k,
            training=is_training,
        )

        # reshape back to (seq_len, d_model)
        x = jnp.transpose(x, (0, 2, 1, 3)).reshape(x.shape[0], -1, self.d_model)
        x = self.w_o(x)
        return x
