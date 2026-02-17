from flax import nnx
from jax import Array


class PatchEmbedding(nnx.Module):
    def __init__(
        self, patch_size: int, in_channels: int, d_model: int, rngs: nnx.Rngs
    ) -> None:
        """
        Create the patch embedding layer. This is a convolutional that will
        project the image into n patches of size patch_size

        Args:
            patch_size: size of the patch
            d_model: dimension of the model
            rngs: nnx.Rngs

        Returns:
            None
        """

        # project the image into patches of size patch_size
        # each conv/patch will learn a representation of the image
        self.projection = nnx.Conv(
            in_features=in_channels,
            out_features=d_model,
            kernel_size=(patch_size, patch_size),
            strides=patch_size,
            padding="VALID",
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        """
        Projects the image into patches

        Args:
            x: image

        Returns:
            Array
        """
        x = self.projection(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        # returns (batch_size, num_patches, d_model)
        return x
