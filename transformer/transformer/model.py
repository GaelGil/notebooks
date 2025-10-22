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
    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1):
        """
        Args:
            d_model: embedding dimension
            seq_len: maximum input sequence length
            dropout: dropout rate (used during training)
        """
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nnx.Dropout(rate=dropout)

        # create positon vector
        # for example, if seq_len = 5, then position = [0, 1, 2, 3, 4]
        position = jnp.arange(seq_len)[:, None]  # (seq_len, 1)

        # create a vector of size d_model/2
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-jnp.log(10000.0) / d_model))

        # create a matrix of size (seq_len, d_model) which is
        # the same as embeddings and fill with zeros
        pe = jnp.zeros((seq_len, d_model))

        # apply sin and cos to even and odd indices in pe
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        # Register as constant buffer (not learned)
        self.pe = pe[None, :, :]  # shape (1, seq_len, d_model)

    def __call__(self, x, *, training: bool):
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)
            training: whether in training mode for dropout
        """
        seq_len = x.shape[1]

        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x, deterministic=not training)


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
        # compute mean and std for each feature of d_model
        # (batch, seq_len, d_model)
        # axis=-1 means along the last dimension which is d_model
        # (batch_size, seq_len, 1) this holds mean of each feature
        mean = jnp.mean(x, axis=-1, keepdims=True)
        # (batch_size, seq_len, 1) holds std of each feature
        std = jnp.std(x, axis=-1, keepdims=True)
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
        self.linear_1 = nnx.Linear(d_model, d_ff, rngs=nnx.Rngs(0))
        self.dropout = nnx.Dropout(rate=dropout)
        self.linear_2 = nnx.Linear(d_ff, d_model, rngs=nnx.Rngs(0))

    def __call__(self, x):
        # simple feed forward network
        # (seq_len, d_model) --> (dff, d_model) --> (seq_len, d_model)
        x = nnx.leaky_relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class MultiHeadAttentionBlock(nnx.Module):
    """
    Multi Head Attention Block
    """

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
        self.w_q = nnx.Linear(d_model, d_model, rngs=nnx.Rngs(0))
        self.w_k = nnx.Linear(d_model, d_model, rngs=nnx.Rngs(0))
        self.w_v = nnx.Linear(d_model, d_model, rngs=nnx.Rngs(0))
        self.w_o = nnx.Linear(d_model, d_model, rngs=nnx.Rngs(0))
        self.dropout = nnx.Dropout(dropout)

    @staticmethod
    def scaled_dot_product_attention(
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask,
        dropout: nnx.Dropout,
    ) -> jnp.ndarray:
        d_k = query.shape[-1]  # get dimension of last axis
        # (Q * K^T)/sqrt(d_k)
        attention_scores = jnp.matmul(query, key.transpose(-2, -1)) / jnp.sqrt(d_k)
        if mask:
            attention_scores = jnp.where(mask == 0, -1e10, attention_scores)
        # softmax(Q * K^T/sqrt(d_k))
        attention_scores = nnx.softmax(attention_scores, axis=-1)
        if dropout:
            attention_scores = dropout(attention_scores, dropout)
        # (Q * K^T)/sqrt(d_k) * V
        x = jnp.matmul(attention_scores, value)
        return x

    def __call__(self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, mask):
        """

        Args:
            q: query
            k: key
            v: value
            mask: mask

        Returns:
            None
        """
        # (seq_len, d_model) * (d_model, d_model) -> (seq_len, d_model)
        query: jnp.ndarray = self.w_q(q)
        key: jnp.ndarray = self.w_k(k)
        value: jnp.ndarray = self.w_v(v)

        # reshape using .view
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
        x = MultiHeadAttentionBlock.scaled_dot_product_attention(
            query, key, value, self.dropout
        )

        # reshape back to (seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)
        x = self.w_o(x)
        return x


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
        # norms and drop out for the residual connections
        self.dropout = nnx.Dropout(rate=dropout)
        self.norm1 = LayerNorm()
        self.norm2 = LayerNorm()

    def __call__(self, x, src_mask):
        # attention block output
        multi_head_attention_output = self.multi_head_attention_block(
            q=x, k=x, v=x, mask=src_mask
        )

        # add and norm output
        x = self.dropout(self.norm1(multi_head_attention_output + x))

        # pass in new x into feed forward and get output
        feed_forward_output = self.feed_forward_block(x)

        # add and norm ff output
        output = self.dropout(self.norm2(feed_forward_output + x))

        return output


class Encoder(nnx.Module):
    def __init__(self, blocks: nnx.List[EncoderBlock]) -> None:
        """
        Args:
            blocks: list of encoder blocks

        Returns:
            None
        """
        self.blocks = nnx.List(blocks)
        self.norm = LayerNorm()

    def __call__(self, x, mask):
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)


class DecoderBlock(nnx.Module):
    def __init__(
        self,
        masked_multi_head_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        self.masked_multi_head_attention_block = masked_multi_head_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nnx.Dropout(dropout)
        self.norm1 = LayerNorm()
        self.norm2 = LayerNorm()
        self.norm3 = LayerNorm()

    def __call__(self, x, encoder_output, src_mask, target_mask):
        # masked multi head attention block output
        masked_multi_head_attention_output = self.masked_multi_head_attention_block(
            q=x, k=x, v=x, mask=target_mask
        )

        # add and norm the masked multi head attention
        x = self.dropout(self.norm1(masked_multi_head_attention_output + x))

        # cross attention
        cross_attention_output = self.cross_attention_block(
            q=x, k=encoder_output, v=encoder_output, mask=src_mask
        )

        # add and norm the cross attention
        x = self.dropout(self.norm2(cross_attention_output + x))

        # feed forward
        feed_forward_output = self.feed_forward_block(x)

        # final add and norm
        output = self.dropout(self.norm3(feed_forward_output + x))

        return output


class Decoder(nnx.Module):
    def __init__(self, blocks: nnx.List[DecoderBlock]) -> None:
        """
        Args:
            blocks: list of decoder blocks

        Returns:
            None
        """
        self.blocks = nnx.List(blocks)
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
        self.linear = nnx.Linear(d_model, vocab_size, rngs=nnx.Rngs(0))

    def __call__(self, x):
        # (seq_len, d_model) --> (seq_len, vocab_size)
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
