from flax import nnx
from jax import numpy as jnp


class ProjectionLayer(nnx.Module):
    """
    Projection Layerr
    Used to map the output of the decoder to the vocabulary. This gives us the logits

    Attributes:
        linear: nnx.Linear
    """

    def __init__(self, vocab_size: int, d_model: int, rngs: nnx.Rngs) -> None:
        """
        Args:
            d_model: dimension of the model
            vocab_size: size of the vocabulary
            rngs: rngs

        Returns:
            None
        """
        self.linear = nnx.Linear(
            in_features=d_model, out_features=vocab_size, dtype=jnp.float32, rngs=rngs
        )

    def __call__(self, x: jnp.ndarray):
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, vocab_size)
        return self.linear(x)
