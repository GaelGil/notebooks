from flax import nnx
from jax import numpy as jnp
from transformer.Encoder import Encoder, EncoderBlock
from transformer.Decoder import Decoder, DecoderBlock
from transformer.InputEmbeddings import InputEmbeddings
from transformer.PositionalEncoding import PositionalEncoding
from transformer.ProjectionLayer import ProjectionLayer


class Transformer(nnx.Module):
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
            d_model=d_model, vocab_size=src_vocab_size
        )
        self.src_pe = PositionalEncoding(
            d_model=d_model, seq_len=seq_len, dropout_rate=dropout
        )

        self.target_embeddings = InputEmbeddings(
            d_model=d_model, vocab_size=target_vocab_size
        )
        self.target_pe = PositionalEncoding(
            d_model=d_model, seq_len=seq_len, dropout_rate=dropout
        )

        self.encoder = Encoder(
            encoder_blocks=nnx.List(
                [
                    EncoderBlock(
                        d_model=d_model,
                        n_heads=n_heads,
                        d_ff=d_ff,
                        dropout_rate=dropout,
                    )
                    for _ in range(N)
                ],
            ),
            d_model=d_model,
        )

        self.decoder = Decoder(
            decoder_blocks=nnx.List(
                [
                    DecoderBlock(
                        d_model=d_model,
                        n_heads=n_heads,
                        d_ff=d_ff,
                        dropout_rate=dropout,
                    )
                    for _ in range(N)
                ]
            ),
            d_model=d_model,
        )

        self.projection = ProjectionLayer(vocab_size=target_vocab_size)

    def __call__(
        self,
        src: jnp.ndarray,
        src_mask: jnp.ndarray,
        target: jnp.ndarray,
        target_mask: jnp.ndarray,
        is_training: bool,
    ):
        # get the embeddings for the src
        src_embeddings = self.src_embeddings(x=src)
        # apply positional encoding to the src embeddings
        src_pos = self.src_pe(x=src_embeddings, is_training=is_training)

        # get the embeddings for the target
        target_embeddings = self.target_embeddings(x=target)
        # apply positonal encoding to the target embeddings
        target_pos = self.target_pe(x=target_embeddings, is_training=is_training)

        # pass the input embeddings with positinal encoding through the encoder
        encoder_output = self.encoder(x=src_pos, mask=src_mask, is_training=is_training)

        # pass the target input embeddings with positional encoding
        # and the encoder output through the decoder
        decoder_output = self.decoder(
            x=target_pos,
            encoder_output=encoder_output,
            src_mask=src_mask,
            target_mask=target_mask,
            is_training=is_training,
        )

        # project the decoder output into vocab size and get outputs
        output = self.projection(decoder_output)

        # jax.debug.print("encoder_output.shape = {}", encoder_output.shape)
        # jax.debug.print("decoder_output.shape = {}", decoder_output.shape)

        return output
