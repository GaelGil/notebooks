from flax import nnx


class EncoderBlock(nnx.Module):
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
        )
        # and one feed forward block
        self.feed_forward_block = FeedForwardBlock(
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout_rate=self.dropout_rate,
        )
        # Lastly there are two residual connections in the encoder block
        # that connect the multi head attention block and the feed forward block
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.norm1 = LayerNorm(d_model=self.d_model)
        self.norm2 = LayerNorm(d_model=self.d_model)

    def __call__(self, x: jnp.ndarray, src_mask: jnp.ndarray, is_training: bool):
        # attention block output
        multi_head_attention_output = self.multi_head_attention_block(
            q=x, k=x, v=x, mask=src_mask, is_training=is_training
        )

        # add and norm output
        x = self.dropout(
            self.norm1(multi_head_attention_output + x), deterministic=not is_training
        )

        # pass in new x into feed forward and get output
        feed_forward_output = self.feed_forward_block(x, is_training=is_training)

        # add and norm ff output
        output = self.dropout(
            self.norm2(feed_forward_output + x), deterministic=not is_training
        )

        return output


class Encoder(nnx.Module):
    encoder_blocks: list[EncoderBlock]
    d_model: int

    def setup(self) -> None:
        """
        Args:
            blocks: list of encoder blocks

        Returns:
            None
        """
        self.blocks: list[EncoderBlock] = self.encoder_blocks
        self.norm: LayerNorm = LayerNorm(d_model=self.d_model)

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray, is_training: bool):
        for block in self.blocks:
            x = block(x=x, src_mask=mask, is_training=is_training)
        return self.norm(x)


class DecoderBlock(nnx.Module):
    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float

    def setup(
        self,
    ) -> None:
        self.masked_multi_head_attention_block = MultiHeadAttentionBlock(
            d_model=self.d_model, n_heads=self.n_heads, dropout_rate=self.dropout_rate
        )
        self.cross_attention_block = MultiHeadAttentionBlock(
            d_model=self.d_model, n_heads=self.n_heads, dropout_rate=self.dropout_rate
        )
        self.feed_forward_block = FeedForwardBlock(
            d_model=self.d_model, d_ff=self.d_ff, dropout_rate=self.dropout_rate
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm1 = LayerNorm(d_model=self.d_model)
        self.norm2 = LayerNorm(d_model=self.d_model)
        self.norm3 = LayerNorm(d_model=self.d_model)

    def __call__(
        self,
        x: jnp.ndarray,
        encoder_output: jnp.ndarray,
        src_mask: jnp.ndarray,
        target_mask: jnp.ndarray,
        is_training: bool,
    ):
        # masked multi head attention block output
        masked_multi_head_attention_output = self.masked_multi_head_attention_block(
            q=x, k=x, v=x, mask=target_mask, is_training=is_training
        )

        # add and norm the masked multi head attention
        x = self.dropout(
            self.norm1(masked_multi_head_attention_output + x),
            deterministic=not is_training,
        )

        # cross attention
        cross_attention_output = self.cross_attention_block(
            q=x,
            k=encoder_output,
            v=encoder_output,
            mask=src_mask,
            is_training=is_training,
        )

        # add and norm the cross attention
        x = self.dropout(
            self.norm2(cross_attention_output + x), deterministic=not is_training
        )

        # feed forward
        feed_forward_output = self.feed_forward_block(x, is_training=is_training)

        # final add and norm
        output = self.dropout(
            self.norm3(feed_forward_output + x), deterministic=not is_training
        )

        return output


class Decoder(nnx.Module):
    decoder_blocks: list[DecoderBlock]
    d_model: int

    def setup(self) -> None:
        """
        Args:
            blocks: list of decoder blocks

        Returns:
            None
        """
        self.blocks: list[DecoderBlock] = self.decoder_blocks
        self.norm: LayerNorm = LayerNorm(d_model=self.d_model)

    def __call__(
        self,
        x: jnp.ndarray,
        encoder_output: jnp.ndarray,
        src_mask: jnp.ndarray,
        target_mask: jnp.ndarray,
        is_training: bool,
    ):
        for block in self.blocks:
            x = block(
                x=x,
                encoder_output=encoder_output,
                src_mask=src_mask,
                target_mask=target_mask,
                is_training=is_training,
            )
        return self.norm(x)


class Transformer(nnx.Module):
    d_model: int
    N: int
    n_heads: int
    d_ff: int
    dropout: float
    seq_len: int
    src_vocab_size: int
    target_vocab_size: int

    def setup(
        self,
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
            d_model=self.d_model, vocab_size=self.src_vocab_size
        )
        self.src_pe = PositionalEncoding(
            d_model=self.d_model, seq_len=self.seq_len, dropout_rate=self.dropout
        )

        self.target_embeddings = InputEmbeddings(
            d_model=self.d_model, vocab_size=self.target_vocab_size
        )
        self.target_pe = PositionalEncoding(
            d_model=self.d_model, seq_len=self.seq_len, dropout_rate=self.dropout
        )

        self.encoder = Encoder(
            encoder_blocks=[
                EncoderBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_ff=self.d_ff,
                    dropout_rate=self.dropout,
                )
                for _ in range(self.N)
            ],
            d_model=self.d_model,
        )

        self.decoder = Decoder(
            decoder_blocks=[
                DecoderBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_ff=self.d_ff,
                    dropout_rate=self.dropout,
                )
                for _ in range(self.N)
            ],
            d_model=self.d_model,
        )

        self.projection = ProjectionLayer(vocab_size=self.target_vocab_size)

    def __call__(
        self,
        src: jnp.ndarray,
        src_mask: jnp.ndarray,
        target: jnp.ndarray,
        target_mask: jnp.ndarray,
        is_training: bool,
    ):
        # get the embeddings for the src
        src_embeddings = self.src_embeddings(x=src)
        # apply positional encoding to the src embeddings
        src_pos = self.src_pe(x=src_embeddings, is_training=is_training)

        # get the embeddings for the target
        target_embeddings = self.target_embeddings(x=target)
        # apply positonal encoding to the target embeddings
        target_pos = self.target_pe(x=target_embeddings, is_training=is_training)

        # pass the input embeddings with positinal encoding through the encoder
        encoder_output = self.encoder(x=src_pos, mask=src_mask, is_training=is_training)

        # pass the target input embeddings with positional encoding
        # and the encoder output through the decoder
        decoder_output = self.decoder(
            x=target_pos,
            encoder_output=encoder_output,
            src_mask=src_mask,
            target_mask=target_mask,
            is_training=is_training,
        )

        # project the decoder output into vocab size and get outputs
        output = self.projection(decoder_output)

        # jax.debug.print("encoder_output.shape = {}", encoder_output.shape)
        # jax.debug.print("decoder_output.shape = {}", decoder_output.shape)

        return output
