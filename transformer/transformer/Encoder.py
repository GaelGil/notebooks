from flax import nnx
from jax import Array

from transformer.AttentionBlock import MultiHeadAttentionBlock
from transformer.FeedForwardBlock import FeedForwardBlock


class EncoderBlock(nnx.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ) -> None:
        """
        Set up encoder block

        Args:
            d_model: dimension of the model
            n_heads: number of heads
            d_ff: dimension of the feed forward network
            dropout_rate: dropout probability
            rngs: rngs

        Returns:
            None
        """
        # encoder block has one self attention block
        self.multi_head_attention_block = MultiHeadAttentionBlock(
            d_model=d_model,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )
        # and one feed forward block
        self.feed_forward_block = FeedForwardBlock(
            d_model=d_model,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )
        # Lastly there are two residual connections in the encoder block
        # that connect the multi head attention block and the feed forward block
        self.dropout = nnx.Dropout(rate=dropout_rate)
        self.norm1 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_features=d_model, rngs=rngs)

    def __call__(
        self, x: Array, src_mask: Array, is_training: bool, rngs: nnx.Rngs
    ) -> Array:
        """
        Args:
            x: input
            src_mask: source mask
            is_training: is training

        Returns:
            Array
        """

        x_norm = self.norm1(x)
        x = x + self.dropout(
            self.multi_head_attention_block(
                q=x_norm,
                k=x_norm,
                v=x_norm,
                mask=src_mask,
                is_training=is_training,
                rngs=rngs,
            ),
            deterministic=not is_training,
            rngs=rngs,
        )

        x_norm = self.norm2(x)
        x = x + self.dropout(
            self.feed_forward_block(x_norm, is_training=is_training, rngs=rngs),
            deterministic=not is_training,
            rngs=rngs,
        )

        return x


class Encoder(nnx.Module):
    def __init__(
        self, encoder_blocks: nnx.List[EncoderBlock], d_model: int, rngs: nnx.Rngs
    ) -> None:
        """
        Args:
            blocks: list of encoder blocks
            d_model: dimension of the model
            rngs: rngs

        Returns:
            None
        """
        self.blocks: nnx.List[EncoderBlock] = encoder_blocks
        self.norm: nnx.LayerNorm = nnx.LayerNorm(num_features=d_model, rngs=rngs)

    def __call__(
        self, x: Array, mask: Array, is_training: bool, rngs: nnx.Rngs
    ) -> Array:
        """
        Args:
            x: input
            mask: mask
            is_training: is training
            rngs: rngs

        Returns:
            Array
        """
        for block in self.blocks:
            x = block(x=x, src_mask=mask, is_training=is_training, rngs=rngs)
        return self.norm(x)
