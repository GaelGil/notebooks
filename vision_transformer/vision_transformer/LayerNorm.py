from flax import nnx
from jax import Array
from jax import numpy as jnp


class LayerNorm(nnx.Module):
    def __init__(self, d_model: int, rngs: nnx.Rngs, eps: float = 1e-6) -> None:
        """Set up layer norm
        Args:
            None

        Returns:
            None
        """
        # create alpha and bias of shape (d_model)
        # alpha and bias are learnable parameters
        # alpha and bias are applied to each patch
        ones = nnx.initializers.ones
        zeros = nnx.initializers.zeros
        self.alpha = nnx.Param(ones(rngs.params(), (d_model,)))
        self.bias = nnx.Param(zeros(rngs.params(), (d_model,)))
        self.eps = eps

    def __call__(self, x: Array):
        # compute mean and std for each patch in the sequence
        # (batch, seq_len, d_model)
        # axis=-1 means along the last dimension which is d_model
        # (batch_size, seq_len, 1) this holds mean of each token in the sequence
        mean = jnp.mean(x, axis=-1, keepdims=True)
        # (batch_size, seq_len, 1) var of each token in the sequence
        var = jnp.var(x, axis=-1, keepdims=True)
        # all elements in x are normalized by mean and std
        return (self.alpha * (x - mean) / jnp.sqrt(var + self.eps)) + self.bias
