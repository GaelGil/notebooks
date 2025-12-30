from transformer.AttentionBlock import MultiHeadAttentionBlock
from transformer.FeedForwardBlock import FeedForwardBlock
from flax import nnx
from jax import Array


class EncoderBlock(nnx.Module):
    """
    Decoder block

    Attributes:
        multi_head_attention_block: MultiHeadAttentionBlock
        norm1: nnx.LayerNorm
        feed_forward_block: FeedForwardBlock
        norm2: nnx.LayerNorm
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
        Set up encoder block

        Args:
            d_model: dimension of the model
            n_heads: number of heads
            d_ff: dimension of the feed forward network
            dropout: dropout probability
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
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.norm1 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_features=d_model, rngs=rngs)

    def __call__(self, x: Array, src_mask: Array, is_training: bool, rngs: nnx.Rngs):
        """
        Args:
            x: input
            src_mask: source mask
            is_training: is training
            rngs: rngs

        Returns:
            Array
        """
        # attention block output
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        multi_head_attention_output = self.multi_head_attention_block(
            q=x, k=x, v=x, mask=src_mask, is_training=is_training, rngs=rngs
        )

        # add and norm output
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        x = self.dropout(
            self.norm1(multi_head_attention_output + x),
            deterministic=not is_training,
            rngs=rngs,
        )

        # pass in new x into feed forward and get output
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        feed_forward_output = self.feed_forward_block(
            x, is_training=is_training, rngs=rngs
        )

        # add and norm ff output
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        output = self.dropout(
            self.norm2(feed_forward_output + x),
            deterministic=not is_training,
            rngs=rngs,
        )

        return output


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

    def __call__(self, x: Array, mask: Array, is_training: bool, rngs: nnx.Rngs):
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
