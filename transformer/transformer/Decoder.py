from flax import nnx
from jax import Array

from transformer.AttentionBlock import MultiHeadAttentionBlock
from transformer.FeedForwardBlock import FeedForwardBlock


class DecoderBlock(nnx.Module):
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
    ) -> Array:
        """
        Args:
            x: input
            encoder_output: encoder output
            self_mask: self mask
            cross_mask: cross mask
            is_training: is training
            rngs: rngs

        Returns:
            Array
        """

        x_norm = self.norm1(x)

        x = x + self.dropout(
            self.masked_multi_head_attention_block(
                q=x_norm,
                k=x_norm,
                v=x_norm,
                mask=self_mask,
                is_training=is_training,
                rngs=rngs,
            ),
            deterministic=not is_training,
            rngs=rngs,
        )

        x_norm = self.norm2(x)

        x = x + self.dropout(
            self.cross_attention_block(
                q=x_norm,
                k=encoder_output,
                v=encoder_output,
                mask=cross_mask,
                is_training=is_training,
                rngs=rngs,
            ),
            deterministic=not is_training,
            rngs=rngs,
        )

        x_norm = self.norm3(x)

        x = x + self.dropout(
            self.feed_forward_block(x_norm, is_training=is_training, rngs=rngs),
            deterministic=not is_training,
            rngs=rngs,
        )

        return x


class Decoder(nnx.Module):
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
    ) -> Array:
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
