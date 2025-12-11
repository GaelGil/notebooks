from transformer.AttentionBlock import MultiHeadAttentionBlock
from transformer.FeedForwardBlock import FeedForwardBlock
from flax import nnx
from jax import numpy as jnp


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
        self.dropout = nnx.Dropout(self.dropout_rate)
        self.norm1 = nnx.LayerNorm(d_model=self.d_model)
        self.norm2 = nnx.LayerNorm(d_model=self.d_model)
        self.norm3 = nnx.LayerNorm(d_model=self.d_model)

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
        self.norm: nnx.LayerNorm = nnx.LayerNorm(d_model=self.d_model)

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
