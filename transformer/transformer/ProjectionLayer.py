from flax import nnx
from jax import numpy as jnp


class ProjectionLayer(nnx.Module):
    """
    Projection layer to map the output of the decoder to the vocabulary. This gives us the logits

    Args:
        d_model: dimension of the model
        vocab_size: size of the vocabulary

    Returns:
        None
    """

    def __init__(self, vocab_size: int, d_model: int, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(
            in_features=d_model, out_features=vocab_size, dtype=jnp.float32, rngs=rngs
        )

    def __call__(self, x: jnp.ndarray):
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, vocab_size)
        return self.linear(x)
