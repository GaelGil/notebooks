from flax import nnx
from jax import Array


class PatchEmbedding(nnx.Module):
    """
    Projects the image into patches

    Attributes:
        patch_size: size of the patch
        d_model: dimension of the model

    """

    def __init__(self, patch_size: int, d_model: int, rngs: nnx.Rngs) -> None:
        """
        Create the patch embedding layer. This is a convolutional that will
        project the image into n patches of size patch_size

        Args:
            None
        Returns:
            None
        """

        # project the image into patches of size patch_size
        # each conv/patch will learn a representation of the image
        self.projection = nnx.Conv(
            in_features=d_model,
            out_features=d_model,
            kernel_size=(patch_size, patch_size),
            strides=patch_size,
            padding="VALID",
            rngs=rngs,
        )

    def __call__(self, x: Array):
        """
        Projects the image into patches

        Args:
            x: image

        Returns:
            x projected into patches
        """
        x = self.projection(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        # returns (batch_size, num_patches, d_model)
        return x
