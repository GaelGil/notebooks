from transformer.model import Transformer, PositionalEncoding, InputEmbeddings, ProjectionLayer, Encoder, Decoder, EncoderBlock, DecoderBlock, MultiHeadAttentionBlock, FeedForwardBlock, ResidualConnection, LayerNorm, PositionalEncoding

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
):
    # create src and target embeddings 
    src_embedding = InputEmbeddings(d_model, src_vocab_size)
    target_embedding = InputEmbeddings(d_model, target_vocab_size)

    # create the position encodings
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    target_pos = PositionalEncoding(d_model, target_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):


