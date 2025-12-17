from flax import nnx
from jax import numpy as jnp

from transformer.Decoder import Decoder, DecoderBlock
from transformer.Encoder import Encoder, EncoderBlock
from transformer.InputEmbeddings import InputEmbeddings
from transformer.PositionalEncoding import PositionalEncoding
from transformer.ProjectionLayer import ProjectionLayer


class Transformer(nnx.Module):
    """
    Transformer model
    Attributes:
        src_embeddings: InputEmbeddings
        src_pe: PositionalEncoding
        target_embeddings: InputEmbeddings
        target_pe: PositionalEncoding
        encoder: Encoder
        decoder: Decoder
        projection_layer: ProjectionLayer
    """

    def __init__(
        self,
        d_model: int,
        N: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        seq_len: int,
        src_vocab_size: int,
        target_vocab_size: int,
        rngs: nnx.Rngs,
    ) -> None:
        """
        Initialize the Transformer model

        Args:
            d_model: The model dimension
            N: number of encoder and decoder blocks
            n_heads: number of heads
            d_ff: The feed forward dimension
            dropout: The dropout probability
            seq_len: The sequence length
            vocab_size: The vocab size

        Return:

            None

        """
        self.src_embeddings = InputEmbeddings(
            d_model=d_model, vocab_size=src_vocab_size, rngs=rngs
        )
        self.src_pe = PositionalEncoding(
            d_model=d_model, seq_len=seq_len, dropout_rate=dropout, rngs=rngs
        )

        self.target_embeddings = InputEmbeddings(
            d_model=d_model, vocab_size=target_vocab_size, rngs=rngs
        )
        self.target_pe = PositionalEncoding(
            d_model=d_model, seq_len=seq_len, dropout_rate=dropout, rngs=rngs
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
                ],
            ),
            d_model=d_model,
            rngs=rngs,
        )

        self.decoder = Decoder(
            decoder_blocks=nnx.List(
                [
                    DecoderBlock(
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

        self.projection = ProjectionLayer(
            vocab_size=target_vocab_size, d_model=d_model, rngs=rngs
        )

    def __call__(
        self,
        src: jnp.ndarray,
        src_mask: jnp.ndarray,
        target: jnp.ndarray,
        self_mask: jnp.ndarray,
        cross_mask: jnp.ndarray,
        is_training: bool,
        rngs: nnx.Rngs,
    ):
        # get the embeddings for the src
        src_embeddings = self.src_embeddings(x=src)
        # apply positional encoding to the src embeddings
        src_pos = self.src_pe(x=src_embeddings, is_training=is_training, rngs=rngs)

        # get the embeddings for the target
        target_embeddings = self.target_embeddings(x=target)
        # apply positonal encoding to the target embeddings
        target_pos = self.target_pe(
            x=target_embeddings, is_training=is_training, rngs=rngs
        )

        # pass the input embeddings with positinal encoding through the encoder
        encoder_output = self.encoder(
            x=src_pos, mask=src_mask, is_training=is_training, rngs=rngs
        )

        # pass the target input embeddings with positional encoding
        # and the encoder output through the decoder
        decoder_output = self.decoder(
            x=target_pos,
            encoder_output=encoder_output,
            self_mask=self_mask,
            cross_mask=cross_mask,
            is_training=is_training,
            rngs=rngs,
        )

        # project the decoder output into vocab size and get outputs
        output = self.projection(decoder_output)

        return output
