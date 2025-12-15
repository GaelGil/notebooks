from flax import nnx
from jax import numpy as jnp

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
        self.masked_multi_head_attention_block = MultiHeadAttentionBlock(
            d_model=d_model, n_heads=n_heads, dropout_rate=dropout_rate, rngs=rngs
        )
        self.cross_attention_block = MultiHeadAttentionBlock(
            d_model=d_model, n_heads=n_heads, dropout_rate=dropout_rate, rngs=rngs
        )
        self.feed_forward_block = FeedForwardBlock(
            d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate, rngs=rngs
        )
        self.dropout = nnx.Dropout(dropout_rate)
        self.norm1 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.norm3 = nnx.LayerNorm(num_features=d_model, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        encoder_output: jnp.ndarray,
        self_mask: jnp.ndarray,
        cross_mask: jnp.ndarray,
        is_training: bool,
    ):
        # masked multi head attention block output
        masked_multi_head_attention_output = self.masked_multi_head_attention_block(
            q=x, k=x, v=x, mask=self_mask, is_training=is_training
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
            mask=cross_mask,
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
    def __init__(
        self, decoder_blocks: nnx.List[DecoderBlock], d_model: int, rngs: nnx.Rngs
    ) -> None:
        """
        Args:
            blocks: list of decoder blocks

        Returns:
            None
        """
        self.blocks: nnx.List[DecoderBlock] = decoder_blocks
        self.norm: nnx.LayerNorm = nnx.LayerNorm(num_features=d_model, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        encoder_output: jnp.ndarray,
        self_mask: jnp.ndarray,
        cross_mask: jnp.ndarray,
        is_training: bool,
    ):
        for block in self.blocks:
            x = block(
                x=x,
                encoder_output=encoder_output,
                self_mask=self_mask,
                cross_mask=cross_mask,
                is_training=is_training,
            )
        return self.norm(x)
