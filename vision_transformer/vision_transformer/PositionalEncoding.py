import jax.numpy as jnp
from flax import nnx
from jax import Array


class PositionalEncoding(nnx.Module):
    """
    Adds positional encoding to the patches

    Attributes:
        d_model: dimension of the model
        num_patches: number of patches
        training: whether in training mode
        dropout_rate: dropout rate
    """

    def __init__(
        self, d_model: int, num_patches: int, dropout_rate: float, rngs: nnx.Rngs
    ):
        """
        Create the positional encoding layer. This is a learnable parameter that we add to the sequence.
        We also create a cls token that we append to the beggining of the sequence.
        Args:
            None

        Returns:
            None
        """
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

        init = nnx.initializers.truncated_normal(stddev=0.02)
        # create cls token with shape (1, 1, d_model)
        # this token is a embedding and will be added to the num patches
        self.cls_token = nnx.Param(init(rngs.params(), (1, 1, d_model)))
        # create positional encoding matrix with shape (1, num_patches + 1, d_model)
        # Each row in the positional encoding matrix is a vector that is added to a
        # corresponding patch
        self.pe = nnx.Param(init(rngs.params(), (1, num_patches + 1, d_model)))

    def __call__(
        self,
        x: Array,
        is_training: bool = True,
        rngs: nnx.Rngs | None = None,
    ):
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)
            training: whether in training mode for dropout
        """
        # get batch size
        B = x.shape[0]

        # duplicate cls token B times so each batch has its own cls token
        cls = jnp.tile(self.cls_token, (B, 1, 1))
        # concatenate cls token to sequence (batch_size, seq_len + 1, d_model)
        x = jnp.concatenate([cls, x], axis=1)

        # add positional encoding to sequence
        x = x + self.pe

        # apply dropout
        # returns (batch_size, num_patches + 1, d_model)
        return self.dropout(x, deterministic=not is_training, rngs=rngs)
