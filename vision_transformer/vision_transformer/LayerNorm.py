from flax import nnx
from jax import numpy as jnp


class LayerNorm(nnx.Module):
    def __init__(self, d_model, eps: float = 1e-6) -> None:
        """Set up layer norm
        Args:
            d_model: dimension of model
            eps: epsilon for numerical stability (avoid division by zero)

        Returns:
            None
        """
        # create alpha and bias of shape (d_model)
        # alpha and bias are learnable parameters
        # alpha and bias are applied to each patch
        self.alpha = self.param("alpha", nnx.initializers.ones, (self.d_model))
        self.bias = self.param("bias", nnx.initializers.zeros, (self.d_model))

    def __call__(self, x: jnp.ndarray):
        # compute mean and std for each patch in the sequence
        # (batch, num_patches, d_model)
        # axis=-1 means along the last dimension which is d_model
        # (batch_size, num_patches, 1) this holds mean of each feature
        mean = jnp.mean(x, axis=-1, keepdims=True)
        # (batch_size, num_patches, 1) holds std of each feature
        var = jnp.var(x, axis=-1, keepdims=True)
        # all elements in x are normalized by mean and std
        return (self.alpha * (x - mean) / jnp.sqrt(var + self.eps)) + self.bias
