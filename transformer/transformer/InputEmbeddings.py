from flax import nnx
from jax import Array


class InputEmbeddings(nnx.Module):
    """
    Input embeddings
    Attributes:
        embedding: nnx.Embed
        d_model: dimension of the model
    """

    def __init__(self, d_model: int, vocab_size: int, rngs: nnx.Rngs) -> None:
        """
        Initialize the embeddings matrix1
        This is a (vocab_size x d_model) matrix so that each token is represented
        by a vector of dimension d_model. These are learned.
        Args:
            d_model: dimension of the model
            vocab_size: size of the vocabulary (num tokens)

        Returns:
            None
        """

        self.embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=d_model,
            embedding_init=nnx.initializers.normal(stddev=d_model**-0.5),
            rngs=rngs,
        )

        self.d_model = d_model

    def __call__(self, x: Array):
        # Get the embedding for each word in x
        # multiply by the square root of d_model for normalization and stability during training
        # (batch_size, seq_len) --> (batch_size, seq_len, d_model
        return self.embedding(x) * self.d_model**0.5
