from flax import nnx


class InputEmbeddings(nnx.Module):
    def __init__(self, d_model: int, vocab_size: int, rngs: nnx.Rngs) -> None:
        """
        Args:
            d_model: dimension of the model
            vocab_size: size of the vocabulary (num tokens)

        Returns:
            None
        """

        # create embeddings matrix.
        # This is a (vocab_size x d_model) matrix so
        # that each word is represented by a vector of dimension d_model.
        # These are learned.

        self.embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=d_model,
            embedding_init=nnx.initializers.normal(stddev=d_model**-0.5),
            rngs=rngs,
        )

        self.d_model = d_model

    def __call__(self, x):
        # Get the embedding for each word in x
        # multiply by the square root of d_model for normalization and stability during training
        return self.embedding(x) * self.d_model**0.5
