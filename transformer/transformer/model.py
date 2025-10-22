import jax.numpy as jnp
from flax import nnx


class InputEmbeddings(nnx.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Args:
            d_model: dimension of the model
            vocab_size: size of the vocabulary (num tokens)

        Returns:
            None
        """
        self.d_model = d_model
        self.vocab_size = vocab_size
        # create embeddings matrix.
        # This is a (vocab_size x d_model) matrix so
        # that each word is represented by a vector of dimension d_model.
        # These are learned.
        self.embedding = nnx.Embed(num_embeddings=vocab_size, num_features=d_model)

    def __call__(self, x):
        # Get the embedding for each word in x
        # multiply by the square root of d_model for normalization and stability during training
        return self.embedding(x) * self.d_model**0.5


class PositionalEncoding(nnx.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout
        # TODO: updated positional encoding
        self.pe = jnp.zeros((seq_len, d_model))
        self.pos = jnp.arange(0, seq_len)[:, jnp.newaxis]
        self.scale = jnp.ones((seq_len, 1))

    def __call__(self, x):
        return x


class LayerNorm(nnx.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        """
        Args:
            eps: epsilon for numerical stability

        Returns:
            None
        """
        self.eps = eps  # helps avoid division by zero
        self.alpha = nnx.Param(jnp.ones(1))
        self.bias = nnx.Param(jnp.zeros(1))

    def __call__(self, x):
        # calculate mean and variance of x
        mean = jnp.mean(x, axis=-1, keepdims=True)
        std = jnp.std(x, axis=-1, keepdims=True)
        # TODO: understand how layernorm is applied
        # all elements in x are normalized by mean and std
        return (self.alpha * (x - mean) / (std + self.eps) ** 0.5) + self.bias


class FeedForwardBlock(nnx.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
        Args:
            d_model: dimension of the model
            d_ff: dimension of the feed forward network
            dropout: dropout probability

        Returns:
            None
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.linear_1 = nnx.Linear(d_model, d_ff)
        self.dropout_1 = nnx.Dropout(dropout)
        self.linear_2 = nnx.Linear(d_ff, d_model)

    def __call__(self, x):
        # simple feed forward network
        # (seq_len, d_model) --> (dff, d_model) --> (seq_len, d_model)
        x = nnx.leaky_relu(self.linear_1(x))
        x = self.dropout_1(x)
        x = self.linear_2(x)
        return x


class MultiHeadAttentionBlock(nnx.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        """

        Args:
            d_model: dimension of the model
            n_heads: number of heads
            dropout: dropout probability

        Returns:
            None
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.w_q = nnx.Linear(d_model, d_model)
        self.w_k = nnx.Linear(d_model, d_model)
        self.w_v = nnx.Linear(d_model, d_model)
        self.w_o = nnx.Linear(d_model, d_model)
        self.dropout = nnx.Dropout(dropout)

    @staticmethod
    def scaled_dot_product_attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        attention_scores = jnp.matmul(query, key.transpose(-2, -1)) / jnp.sqrt(d_k)
        if mask is not None:
            # TODO: understand mask
            attention_scores = attention_scores + mask
        attention_weights = nnx.softmax(attention_scores, axis=-1)
        if dropout is not None:
            attention_weights = nnx.dropout(attention_weights, dropout)
        x = jnp.matmul(attention_weights, value)
        return x, attention_weights

    def __call__(self, q, k, v, mask):
        # (seq_len, d_model) * (d_model, d_model) -> (seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # reshape using .view
        # (seq_len, d_model) -> (seq_len, n_heads, d_k) -> (n_heads, seq_len, d_k)
        # where dk = d_model // n_heads
        # (n, 512) -> (n, h, dk) -> (h, n, dk)
        # (3, 512) -> (3, 8, 64) -> (8, 3, 64)
        #
        # Sequence length n where each token is of dimension 512 ->
        # Sequence length n where each token is an array of 8 vectors of dimension 64 ->
        # Explaination: The embeddings are split into 8 parts so that each head can focus on different parts of the embeddings
        # 8 Heads where each head contains a matrix of n vectors of dimension 64
        query = query.view(
            query.shape[0], query.shape[1], self.n_heads, self.d_k
        ).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(
            1, 2
        )
        value = value.view(
            value.shape[0], value.shape[1], self.n_heads, self.d_k
        ).transpose(1, 2)

        # apply scaled dot product attention to each head
        x, self.attention_scores = MultiHeadAttentionBlock.scaled_dot_product_attention(
            query, key, value, self.dropout
        )

        # reshape back to (seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)
        x = self.w_o(x)
        return x


class ResidualConnection(nnx.Module):
    def __init__(self, dropout: float) -> None:
        self.dropout = nnx.Dropout(dropout)
        self.layer_norm = LayerNorm()

    def __call__(self, x, sublayer):
        # residual connection
        # sublayer is either multi head attention block or feed forward block
        # see paper for more details
        return x + self.dropout(sublayer(self.layer_norm(x)))


class EncoderBlock(nnx.Module):
    def __init__(
        self,
        multi_head_attention_block: MultiHeadAttentionBlock,
        feed_forward: FeedForwardBlock,
        dropout: float,
    ) -> None:
        """
        Args:
            multi_head_attention: multi head attention block
            feed_forward: feed forward block
            dropout: dropout probability

        Returns:
            None
        """
        # encoder block has one self attention block
        self.multi_head_attention_block = multi_head_attention_block
        # and one feed forward block
        self.feed_forward_block = feed_forward
        # Lastly there are two residual connections in the encoder block
        # that connect the multi head attention block and the feed forward block
        self.residual_connections: list[nnx.Module] = [
            ResidualConnection(dropout) for _ in range(2)
        ]

    def __call__(self, x, src_mask):
        # as explained in the paper we pass the input embedding into the residual connection which contains the multi head attention block and the add and layer norm
        x = self.residual_connections[0](
            x, lambda x: self.multi_head_attention_block(x, x, x, src_mask)
        )
        # then the output is passed into the feed forward block as well as the residual connection
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nnx.Module):
    def __init__(self, blocks: list[nnx.Module]) -> None:
        """
        Args:
            blocks: list of encoder blocks

        Returns:
            None
        """
        self.blocks = blocks
        self.norm = LayerNorm()

    def __call__(self, x, mask):
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)


class DecoderBlock(nnx.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.cross_attention_block = cross_attention_block
        self.residual_connections: list[nnx.Module] = [
            ResidualConnection(dropout) for _ in range(3)
        ]

    def __call__(self, x, encoder_output, src_mask, target_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, target_mask)
        )

        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )

        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nnx.Module):
    def __init__(self, blocks: list[nnx.Module]) -> None:
        """
        Args:
            blocks: list of decoder blocks

        Returns:
            None
        """
        self.blocks = blocks
        self.norm = LayerNorm()

    def __call__(self, x, encoder_output, src_mask, target_mask):
        for block in self.blocks:
            x = block(x, encoder_output, src_mask, target_mask)
        return self.norm(x)


class ProjectionLayer(nnx.Module):
    """
    Projection layer to map the output of the decoder to the vocabulary. This gives us the logits

    Args:
        d_model: dimension of the model
        vocab_size: size of the vocabulary

    Returns:
        None"""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        self.linear = nnx.Linear(d_model, vocab_size)

    def __call__(self, x):
        # (seq_len, d_model) -> (seq_len, vocab_size)
        return nnx.log_softmax(self.linear(x))


class Transformer(nnx.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embedding: InputEmbeddings,
        target_embedding: InputEmbeddings,
        src_pos: PositionalEncoding,
        target_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.target_embedding = target_embedding
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, target, target_mask):
        target = self.target_embedding(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_output, src_mask, target_mask)

    def projection(self, x):
        return self.projection_layer(x)
