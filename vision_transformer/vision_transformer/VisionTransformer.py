from flax import nnx
from jax import Array

from vision_transformer.Encoder import Encoder, EncoderBlock
from vision_transformer.PatchEmbedding import PatchEmbedding
from vision_transformer.PositionalEncoding import PositionalEncoding
from vision_transformer.ProjectionLayer import ProjectionLayer


class VisionTransformer(nnx.Module):
    def __init__(
        self,
        N: int,
        n_heads: int,
        dropout: float,
        img_size: int,
        patch_size: int,
        in_channels: int,
        d_model: int,
        d_ff: int,
        num_classes: int,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """
        Set up the vision transformer. With a patch embedding, positional encoding, projection layer and encoder
        Args:
            N: number of encoder blocks
            n_heads: number of heads
            dropout: dropout probability
            img_size: image size
            patch_size: patch size
            in_channels: number of channels
            d_model: dimension of the model
            num_classes: the number of classes in our dataset


        Returns:
            None
        """

        self.patch_embedding = PatchEmbedding(
            patch_size=patch_size, in_channels=in_channels, d_model=d_model, rngs=rngs
        )
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            num_patches=(img_size // patch_size) ** 2,
            dropout_rate=dropout,
            rngs=rngs,
        )
        self.projection_layer = ProjectionLayer(
            d_model=d_model, num_classes=num_classes, rngs=rngs
        )

        self.encoder = Encoder(
            encoder_blocks=nnx.List(
                [
                    EncoderBlock(
                        d_model=d_model,
                        n_heads=n_heads,
                        d_ff=d_ff,
                        dropout_rate=dropout,
                        rngs=rngs,
                    )
                    for _ in range(N)
                ]
            ),
            d_model=d_model,
            rngs=rngs,
        )

    def __call__(
        self,
        x: Array,
        is_training: bool,
        rngs: nnx.Rngs | None = None,
    ) -> Array:
        """
        Call the vision transformer
        Args:
            x: input
            is_training: bool

        Returns:
            Array
        """
        x = self.patch_embedding(x=x)
        x = self.positional_encoding(x=x, is_training=is_training, rngs=rngs)
        x = self.encoder(x=x, is_training=is_training, rngs=rngs)
        x = self.projection_layer(x)
        return x
