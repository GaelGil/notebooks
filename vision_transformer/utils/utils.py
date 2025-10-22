from flax import nnx

from vision_transformer.model import (
    Encoder,
    EncoderBlock,
    MultiHeadAttentionBlock,
    MultiLayerPerceptron,
    PatchEmbedding,
    PositionalEncoding,
    VisionTransformer,
)


def build_vision_transformer(
    src_vocab_size: int,
    target_vocab_size: int,
    src_seq_len: int,
    target_seq_len: int,
    patch_size: int,
    in_channels: int = 3,
    num_patches: int = 16,
    img_size: int = 32,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    d_ff: int = 2048,
    dropout: float = 0.1,
) -> VisionTransformer:
    # create src and target embeddings
    patch_embeddings = PatchEmbedding(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        d_model=d_model,
    )

    # create the position encodings
    patch_pos = PositionalEncoding(d_model, src_seq_len, dropout)

    encoder_blocks: nnx.List[EncoderBlock] = []
    for _ in range(N):
        multi_head_attention_block = MultiHeadAttentionBlock(
            d_model=d_model, n_heads=h, dropout=dropout
        )
        multi_layer_perceptron_block = MultiLayerPerceptron(
            d_model=d_model, d_ff=d_ff, dropout=dropout
        )

        encoder_block = EncoderBlock(
            multi_head_attention_block=multi_head_attention_block,
            feed_forward=multi_layer_perceptron_block,
            dropout=dropout,
        )
        encoder_blocks.append(encoder_block)

    encoder = Encoder(encoder_blocks)

    # create the model
    model = VisionTransformer(
        encoder=encoder,
        patch_embedding=patch_embeddings,
        positional_encoding=patch_pos,
        projection_layer=nnx.Linear(
            in_features=target_vocab_size,
            out_features=target_vocab_size,
            rngs=nnx.Rngs(0),
        ),
    )

    return model
