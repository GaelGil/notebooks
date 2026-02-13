from flax import nnx
from jax import Array

from vision_transformer.LayerNorm import LayerNorm
from vision_transformer.MultiHeadAttentionBlock import MultiHeadAttentionBlock
from vision_transformer.MultiLayerPerceptron import MultiLayerPerceptron


class EncoderBlock(nnx.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout_rate: float, rngs: nnx.Rngs
    ) -> None:
        """
        Set up encoder block
        Each encoder block has one multi head attention block and one feed forward block.
        There is also the residual connections of which it has two.
        Args:
            d_model: dimension of model
            n_heads: number of heads in multi head attention block
            d_ff: dimension of feed forward network
            dropout_rate: dropout rate
            rngs: nnx.Rngs

        Returns:
            None
        """
        # encoder block has one self attention block
        self.multi_head_attention_block = MultiHeadAttentionBlock(
            d_model=d_model, n_heads=n_heads, dropout_rate=dropout_rate, rngs=rngs
        )
        # and one feed forward block
        self.multi_layer_perceptron_block = MultiLayerPerceptron(
            d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate, rngs=rngs
        )
        # Lastly there are two residual connections in the encoder block
        # that connect the multi head attention block and the feed forward block
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.norm1 = LayerNorm(d_model=d_model)
        self.norm2 = LayerNorm(d_model=d_model)

    def __call__(
        self,
        x: Array,
        is_training: bool,
        rngs: nnx.Rngs | None = None,
    ):
        multi_head_attention_output = self.multi_head_attention_block(
            q=x, k=x, v=x, is_training=is_training, rngs=rngs
        )

        x = self.dropout(
            self.norm1(multi_head_attention_output + x),
            deterministic=not is_training,
            rngs=rngs,
        )

        multi_layer_perceptron_output = self.multi_layer_perceptron_block(
            x=x, is_training=is_training, rngs=rngs
        )

        output = self.dropout(
            self.norm2(multi_layer_perceptron_output + x),
            deterministic=not is_training,
            rngs=rngs,
        )

        return output


class Encoder(nnx.Module):
    def __init__(self, encoder_blocks: nnx.List[EncoderBlock], d_model: int) -> None:
        """
        Set up encoder with a sequence of encoder blocks
        Args:
            encoder_blocks: A sequence of encoder blocks
            d_model: dimension of model

        Returns:
            None
        """
        self.blocks: nnx.List[EncoderBlock] = encoder_blocks
        self.norm = LayerNorm(d_model=d_model)

    def __call__(
        self,
        x: Array,
        is_training: bool,
        rngs: nnx.Rngs | None = None,
    ) -> Array:
        """
        Call the encoder

        Args:
            x: input
            is_training: bool

        Returns:
            Array
        """
        for block in self.blocks:
            x = block(x, is_training=is_training, rngs=rngs)
        return self.norm(x)
