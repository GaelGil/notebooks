import jax.numpy as jnp
from flax import linen as nn


class InputEmbeddings(nn.Module):
    d_model: int
    vocab_size: int

    def setup(
        self,
    ) -> None:
        """
        Args:
            d_model: dimension of the model
            vocab_size: size of the vocabulary (num tokens)

        Returns:
            None
        """

        # create embeddings matrix.
        # This is a (vocab_size x d_model) matrix so
        # that each word is represented by a vector of dimension d_model.
        # These are learned.
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)

    def __call__(self, x):
        # Get the embedding for each word in x
        # multiply by the square root of d_model for normalization and stability during training
        return self.embedding(x) * self.d_model**0.5


class PositionalEncoding(nn.Module):
    def setup(self, d_model: int, seq_len: int, dropout: float = 0.1):
        """
        Args:
            d_model: embedding dimension
            seq_len: maximum input sequence length
            dropout: dropout rate (used during training)
        """
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(rate=dropout)

        # create positon vector
        # for example, if seq_len = 5, then position = [0, 1, 2, 3, 4]
        # in our case we create a vector of size seq_len
        position = jnp.arange(seq_len)[:, None]  # (seq_len, 1)

        # create a vector of size d_model/2
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-jnp.log(10000.0) / d_model))

        # create a matrix of size (seq_len, d_model) which is
        # the same as embeddings and fill with zeros
        pe = jnp.zeros((seq_len, d_model))

        # apply sin and cos to even and odd indices in pe matrix
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


class LayerNorm(nn.Module):
    eps: float = 1e-6  # helps avoid division by zero

    def setup(self) -> None:
        """Set up layer norm
        Args:
            None

        Returns:
            None
        """
        self.alpha = self.param("alpha", nn.initializers.ones, (1))
        self.bias = self.param("bias", nn.initializers.zeros, (1))

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


class FeedForwardBlock(nn.Module):
    d_model: int
    d_ff: int
    dropout_rate: float
    training: bool

    def setup(self) -> None:
        """
        Args:
            d_model: dimension of the model
            d_ff: dimension of the feed forward network
            dropout: dropout probability

        Returns:
            None
        """

        self.linear_1 = nn.Dense(features=self.d_ff)
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.linear_2 = nn.Dense(features=self.d_model)

    def __call__(self, x):
        # simple feed forward network
        # (seq_len, d_model) --> (dff, d_model) --> (seq_len, d_model)
        x = nn.leaky_relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class MultiHeadAttentionBlock(nn.Module):
    """
    Multi Head Attention Block
    """

    def setup(self, d_model: int, n_heads: int, dropout: float) -> None:
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

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.w_q = nn.Dense(features=d_model)
        self.w_k = nn.Dense(features=d_model)
        self.w_v = nn.Dense(features=d_model)
        self.w_o = nn.Dense(features=d_model)
        self.dropout = nn.Dropout(rate=dropout)

    @staticmethod
    def scaled_dot_product_attention(
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask,
        dropout: nn.Dropout,
    ) -> jnp.ndarray:
        d_k = query.shape[-1]  # get dimension of last axis
        # (Q * K^T)/sqrt(d_k)
        attention_scores = jnp.matmul(query, key.transpose(-2, -1)) / jnp.sqrt(d_k)
        if mask:
            attention_scores = jnp.where(mask == 0, -1e10, attention_scores)
        # softmax(Q * K^T/sqrt(d_k))
        attention_scores = nn.softmax(attention_scores, axis=-1)
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


class EncoderBlock(nn.Module):
    """
    Atttributes:
        d_model: dimension of model
        n_heads: number of heads
        d_ff: dimension of feed forward network
        dropout_rate: dropout rate
        training: whether in training mode
    """

    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float
    training: bool

    def setup(self) -> None:
        """
        Set up encoder block
        Each encoder block has one multi head attention block and one feed forward block.
        There is also the residual connections of which it has two.
        Args:
            None

        Returns:
            None
        """
        # encoder block has one self attention block
        self.multi_head_attention_block = MultiHeadAttentionBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate,
            training=self.training,
        )
        # and one feed forward block
        self.multi_layer_perceptron_block = FeedForwardBlock(
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout_rate=self.dropout_rate,
            training=self.training,
        )
        # Lastly there are two residual connections in the encoder block
        # that connect the multi head attention block and the feed forward block
        self.dropout = nn.Dropout(rate=self.dropout_rate)
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


class Encoder(nn.Module):
    def setup(self, blocks: nn.Sequential) -> None:
        """
        Args:
            blocks: list of encoder blocks

        Returns:
            None
        """
        self.blocks: nn.Sequential = blocks
        self.norm = LayerNorm()

    def __call__(self, x, mask):
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def setup(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        self.masked_multi_head_attention_block = MultiHeadAttentionBlock(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )
        self.cross_attention_block = MultiHeadAttentionBlock(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )
        self.feed_forward_block = FeedForwardBlock(
            d_model=d_model, d_ff=d_ff, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
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


class Decoder(nn.Module):
    def setup(self, blocks: nn.Sequential) -> None:
        """
        Args:
            blocks: list of decoder blocks

        Returns:
            None
        """
        self.blocks: nn.Sequential = blocks
        self.norm = LayerNorm()

    def __call__(self, x, encoder_output, src_mask, target_mask):
        for block in self.blocks:
            x = block(x, encoder_output, src_mask, target_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    """
    Projection layer to map the output of the decoder to the vocabulary. This gives us the logits

    Args:
        d_model: dimension of the model
        vocab_size: size of the vocabulary

    Returns:
        None
    """

    def setup(self, vocab_size: int) -> None:
        self.linear = nn.Dense(features=vocab_size)

    def __call__(self, x):
        # (seq_len, d_model) --> (seq_len, vocab_size)
        return nn.log_softmax(self.linear(x))


class Transformer(nn.Module):
    def setup(
        self,
        d_model: int,
        N: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        seq_len: int,
        src_vocab_size: int,
        target_vocab_size: int,
    ) -> None:
        """
        Initialize the Transformer model

        Args:
            d_model: The model dimension
            N: number of encoder and decoder blocks
            n_heads: number of heads
            d_ff: The feed forward dimension
            dropout: The dropout probability
            seq_len: The sequence length
            vocab_size: The vocab size

        Return:

            None

        """
        self.src_embeddings = InputEmbeddings(
            d_model=d_model, vocab_size=src_vocab_size
        )
        self.src_pe = PositionalEncoding(
            d_model=d_model, seq_len=seq_len, dropout=dropout
        )

        self.target_embeddings = InputEmbeddings(
            d_model=d_model, vocab_size=target_vocab_size
        )
        self.target_pe = PositionalEncoding(
            d_model=d_model, seq_len=seq_len, dropout=dropout
        )

        self.encoder = Encoder(
            nn.Sequential(
                layers=[
                    EncoderBlock(
                        d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout
                    )
                    for _ in range(N)
                ]
            )
        )

        self.decoder = Decoder(
            nn.Sequential(
                layers=[
                    DecoderBlock(
                        d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout
                    )
                    for _ in range(N)
                ]
            )
        )

        self.projection = ProjectionLayer(d_model=d_model, vocab_size=target_vocab_size)

    def __call__(self, src, src_mask, target, target_mask):
        # get the embeddings for the src
        src_embeddings = self.src_embeddings(x=src)
        # apply positional encoding to the src embeddings
        src_pos = self.src_pe(x=src_embeddings)

        # get the embeddings for the target
        target_embeddings = self.target_embeddings(x=target)
        # apply positonal encoding to the target embeddings
        target_pos = self.src_pe(x=target_embeddings)

        # pass the input embeddings with positinal encoding through the encoder
        encoder_output = self.encoder(x=src_pos, mask=src_mask)

        # pass the target input embeddings with positional encoding
        # and the encoder output through the decoder
        decoder_output = self.decoder(
            x=target_pos,
            encoder_output=encoder_output,
            src_mask=src_mask,
            target_mask=target_mask,
        )

        # project the decoder output into vocab size and get outputs
        output = self.projection(decoder_output)

        return output
