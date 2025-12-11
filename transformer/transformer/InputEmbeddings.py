from flax import nnx


class InputEmbeddings(nnx.Module):
    d_model: int
    vocab_size: int

    def setup(
        self,
    ) -> None:
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
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=nnx.initializers.normal(stddev=self.d_model**-0.5),
        )

    def __call__(self, x):
        # Get the embedding for each word in x
        # multiply by the square root of d_model for normalization and stability during training
        return self.embedding(x) * self.d_model**0.5
