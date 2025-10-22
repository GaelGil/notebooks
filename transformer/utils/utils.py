from flax import nnx

from transformer.model import (
    Decoder,
    DecoderBlock,
    Encoder,
    EncoderBlock,
    FeedForwardBlock,
    InputEmbeddings,
    MultiHeadAttentionBlock,
    PositionalEncoding,
    ProjectionLayer,
    Transformer,
)


def build_transformer(
    src_vocab_size: int,
    target_vocab_size: int,
    src_seq_len: int,
    target_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    d_ff: int = 2048,
    dropout: float = 0.1,
) -> Transformer:
    # create src and target embeddings
    src_embedding = InputEmbeddings(d_model, src_vocab_size)
    target_embedding = InputEmbeddings(d_model, target_vocab_size)

    # create the position encodings
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    target_pos = PositionalEncoding(d_model, target_seq_len, dropout)

    # create the encoder with n encoding blocks
    encoder_blocks: nnx.List[EncoderBlock] = nnx.List()
    for _ in range(N):
        encoder_multi_head_attention_block = MultiHeadAttentionBlock(
            d_model=d_model, n_heads=h, dropout=dropout
        )
        encoder_feed_forward_block = FeedForwardBlock(
            d_model=d_model, d_ff=d_ff, dropout=dropout
        )
        encoder_block = EncoderBlock(
            multi_head_attention_block=encoder_multi_head_attention_block,
            feed_forward=encoder_feed_forward_block,
        )
        encoder_blocks.append(encoder_block)

    encoder = Encoder(blocks=encoder_blocks)

    # create the decoder with n decoding blocks
    decoder_blocks: nnx.List[DecoderBlock] = nnx.List()
    for _ in range(N):
        decoder_masked_multi_head_attention_block = MultiHeadAttentionBlock(
            d_model=d_model, n_heads=h, dropout=dropout
        )
        decoder_cross_attention_block = MultiHeadAttentionBlock(
            d_model=d_model, n_heads=h, dropout=dropout
        )

        decoder_feed_forward_block = FeedForwardBlock(
            d_model=d_model, d_ff=d_ff, dropout=dropout
        )
        decoder_block = DecoderBlock(
            masked_multi_head_attention_block=decoder_masked_multi_head_attention_block,
            cross_attention_block=decoder_cross_attention_block,
            feed_forward_block=decoder_feed_forward_block,
        )
        decoder_blocks.append(decoder_block)

    decoder = Decoder(decoder_blocks)

    # create the projection layer
    projection = ProjectionLayer(d_model=d_model, vocab_size=target_vocab_size)

    # put everything together
    transformer = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_embedding=src_embedding,
        target_embedding=target_embedding,
        src_pos=src_pos,
        target_pos=target_pos,
        projection_layer=projection,
    )
    return transformer
