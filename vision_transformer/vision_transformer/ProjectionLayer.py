from flax import nnx
from jax import Array


class ProjectionLayer(nnx.Module):
    """
    Projection layer to map the output of the decoder to the number of classes. This gives us the logits

    Attributes:
        num_classes: the number of classes in our dataset
    """

    def __init__(self, d_model: int, num_classes: int, rngs: nnx.Rngs) -> None:
        """
        Set up projection layer. Should map the output of the encoder to the number of classes in our dataset
        Args:
            None

        Returns:
            None
        """
        self.linear = nnx.Linear(
            in_features=d_model, out_features=num_classes, rngs=rngs
        )

    def __call__(self, x: Array):
        cls_token = x[:, 0]  # shape: (batch_size, d_model) get the cls token
        logits = self.linear(
            cls_token
        )  # shape: (batch_size, num_classes) get the logits
        return logits  # get logits/raw output
