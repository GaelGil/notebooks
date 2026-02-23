from flax import nnx
from jax import Array

from transformer.AttentionBlock import MultiHeadAttentionBlock
from transformer.FeedForwardBlock import FeedForwardBlock


class DecoderBlock(nnx.Module):
    """
    Decoder block
    Attributes:
        masked_multi_head_attention_block: MultiHeadAttentionBlock
        nomr1: nnx.LayerNorm
        cross_attention_block: MultiHeadAttentionBlock
        norm2: nnx.LayerNorm
        feed_forward_block: FeedForwardBlock
        norm3: nnx.LayerNorm

    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ) -> None:
        """
        Set up decoder block
        Args:
            d_model: dimension of the model
            n_heads: number of heads
            d_ff: dimension of the feed forward network
            dropout: dropout probability
            rngs: rngs

        Returns:
            None
        """
        self.masked_multi_head_attention_block = MultiHeadAttentionBlock(
            d_model=d_model, n_heads=n_heads, dropout_rate=dropout_rate, rngs=rngs
        )
        self.cross_attention_block = MultiHeadAttentionBlock(
            d_model=d_model, n_heads=n_heads, dropout_rate=dropout_rate, rngs=rngs
        )
        self.feed_forward_block = FeedForwardBlock(
            d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate, rngs=rngs
        )
        self.dropout = nnx.Dropout(rate=dropout_rate)
        self.norm1 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.norm3 = nnx.LayerNorm(num_features=d_model, rngs=rngs)

    def __call__(
        self,
        x: Array,
        encoder_output: Array,
        self_mask: Array,
        cross_mask: Array,
        is_training: bool,
        rngs: nnx.Rngs,
    ):
        """
        Args:
            x: input
            encoder_output: encoder output
            self_mask: self mask
            cross_mask: cross mask
            is_training: is training

        Returns:
            Array
        """
        # masked multi head attention block output
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        masked_multi_head_attention_output = self.masked_multi_head_attention_block(
            q=x,
            k=x,
            v=x,
            mask=self_mask,
            is_training=is_training,
            rngs=rngs,
        )

        # add and norm the masked multi head attention
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        x = self.dropout(
            self.norm1(masked_multi_head_attention_output + x),
            deterministic=not is_training,
            rngs=rngs,
        )

        # cross attention
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        cross_attention_output = self.cross_attention_block(
            q=x,
            k=encoder_output,
            v=encoder_output,
            mask=cross_mask,
            is_training=is_training,
            rngs=rngs,
        )

        # add and norm the cross attention
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        x = self.dropout(
            self.norm2(cross_attention_output + x),
            deterministic=not is_training,
            rngs=rngs,
        )

        # feed forward
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        feed_forward_output = self.feed_forward_block(
            x, is_training=is_training, rngs=rngs
        )

        # final add and norm
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        x = self.dropout(
            self.norm3(feed_forward_output + x),
            deterministic=not is_training,
            rngs=rngs,
        )

        return x


class Decoder(nnx.Module):
    """
    Decoder
    list of decoder blocks
    layer norm
    """

    def __init__(
        self, decoder_blocks: nnx.List[DecoderBlock], d_model: int, rngs: nnx.Rngs
    ) -> None:
        """
        Args:
            blocks: list of decoder blocks
            d_model: dimension of the model
            rngs: rngs

        Returns:
            None
        """
        self.blocks: nnx.List[DecoderBlock] = decoder_blocks
        self.norm: nnx.LayerNorm = nnx.LayerNorm(num_features=d_model, rngs=rngs)

    def __call__(
        self,
        x: Array,
        encoder_output: Array,
        self_mask: Array,
        cross_mask: Array,
        is_training: bool,
        rngs: nnx.Rngs,
    ):
        """
        Args:
            x: input
            encoder_output: encoder output
            self_mask: self mask
            cross_mask: cross mask
            is_training: is training

        Returns:
            Array
        """
        for block in self.blocks:
            x = block(
                x=x,
                encoder_output=encoder_output,
                self_mask=self_mask,
                cross_mask=cross_mask,
                is_training=is_training,
                rngs=rngs,
            )
        return self.norm(x)
