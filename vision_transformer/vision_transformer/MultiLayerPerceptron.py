from flax import nnx
from jax import Array


class MultiLayerPerceptron(nnx.Module):
    def __init__(
        self, d_model: int, d_ff: int, dropout_rate: float, rngs: nnx.Rngs
    ) -> None:
        """
        Create a feed forward network. With two linear layers.
        It should follow (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        Args:
            d_model: dimension of model
            d_ff: dimension of feed forward network
            dropout_rate: dropout rate
            rngs: nnx.Rngs

        Returns:
            None
        """
        self.linear_1 = nnx.Linear(in_features=d_model, out_features=d_ff, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.linear_2 = nnx.Linear(in_features=d_ff, out_features=d_model, rngs=rngs)

    def __call__(
        self,
        x: Array,
        is_training: bool,
        rngs: nnx.Rngs | None = None,
    ) -> Array:
        """simple feed forward network
        Args:
            x: input to the feed forward network
            is_training: whether we are training or not

        Returns:
            Array
        """
        x = nnx.gelu(self.linear_1(x))
        x = self.dropout(x, deterministic=not is_training, rngs=rngs)
        x = self.linear_2(x)
        return x
