from transformer.AttentionBlock import MultiHeadAttentionBlock
from transformer.FeedForwardBlock import FeedForwardBlock
from flax import nnx
from jax import numpy as jnp


class EncoderBlock(nnx.Module):
    """
    Atttributes:
        d_model: dimension of model
        n_heads: number of heads
        d_ff: dimension of feed forward network
        dropout_rate: dropout rate
        training: whether in training mode
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout_rate: float,
    ) -> None:
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
            d_model=d_model,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
        )
        # and one feed forward block
        self.feed_forward_block = FeedForwardBlock(
            d_model=d_model,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
        )
        # Lastly there are two residual connections in the encoder block
        # that connect the multi head attention block and the feed forward block
        self.dropout = nnx.Dropout(rate=dropout_rate)
        self.norm1 = nnx.LayerNorm(num_features=d_model)
        self.norm2 = nnx.LayerNorm(num_features=d_model)

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
    def __init__(self, encoder_blocks: nnx.List[EncoderBlock], d_model: int) -> None:
        """
        Args:
            blocks: list of encoder blocks

        Returns:
            None
        """
        self.blocks: nnx.List[EncoderBlock] = self.encoder_blocks
        self.norm: nnx.LayerNorm = nnx.LayerNorm(num_features=self.d_model)

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray, is_training: bool):
        for block in self.blocks:
            x = block(x=x, src_mask=mask, is_training=is_training)
        return self.norm(x)
