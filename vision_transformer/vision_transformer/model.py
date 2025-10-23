import jax.numpy as jnp
from flax import nnx


class PatchEmbedding(nnx.Module):
    """
    Projects the image into patches
    """

    def __init__(
        self, img_size: int, patch_size: int, in_channels: int, d_model: int
    ) -> None:
        """
        Args:
            img_size: size of the image
            patch_size: size of the patch
            d_model: dimension of the model


        Returns:
            None
        """
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_patches = (img_size // patch_size) ** 2

        # project the image into patches of size patch_size
        # each conv/patch will learn a representation of the image
        self.projection = nnx.Conv(
            in_features=in_channels,
            out_features=d_model,
            kernel_size=patch_size,
            strides=patch_size,
            rngs=nnx.Rngs(0),
        )

    def __call__(self, x: jnp.ndarray):
        """
        Projects the image into patches

        Args:
            x: image

        Returns:
            None
        """
        x = self.projection(x)
        x = x.flatten(2).transpose(0, 2, 1)

        return x


class PositionalEncoding(nnx.Module):
    def __init__(self, d_model: int, num_patches: int, dropout: float = 0.1):
        """
        Args:
            d_model: embedding dimension
            seq_len: maximum input sequence length
            dropout: dropout rate (used during training)
        """
        self.dropout = nnx.Dropout(rate=dropout)

        self.cls_token = nnx.Param(jnp.zeros((1, 1, d_model)))
        self.pe = nnx.Param(jnp.zeros((1, num_patches + 1, d_model)), rngs=nnx.Rngs(0))

    def __call__(self, x: jnp.ndarray, training: bool):
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)
            training: whether in training mode for dropout
        """
        B = x.shape[0]

        # prepend cls token
        cls = jnp.tile(self.cls_token.value, (B, 1, 1))  # (B, 1, D)
        x = jnp.concatenate([cls, x], axis=1)  # (B, N+1, D)

        # add positional encoding
        x = x + self.pe.value

        return self.dropout(x, deterministic=not training)


class LayerNorm(nnx.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        """
        Args:
            eps: epsilon for numerical stability

        Returns:
            None
        """
        self.eps = eps  # helps avoid division by zero
        self.alpha = nnx.Param(jnp.ones(1))
        self.bias = nnx.Param(jnp.zeros(1))

    def __call__(self, x):
        # compute mean and std for each feature of d_model
        # (batch, seq_len, d_model)
        # axis=-1 means along the last dimension which is d_model
        # (batch_size, seq_len, 1) this holds mean of each feature
        mean = jnp.mean(x, axis=-1, keepdims=True)
        # (batch_size, seq_len, 1) holds std of each feature
        std = jnp.std(x, axis=-1, keepdims=True)
        # all elements in x are normalized by mean and std
        return (self.alpha * (x - mean) / (std + self.eps) ** 0.5) + self.bias


class MultiLayerPerceptron(nnx.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
        Args:
            d_model: dimension of the model
            d_ff: dimension of the feed forward network
            dropout: dropout probability

        Returns:
            None
        """
        self.d_model = d_model
        self.d_ff = d_ff
        # self.dropout = dropout
        self.linear_1 = nnx.Linear(d_model, d_ff, rngs=nnx.Rngs(0))
        self.dropout = nnx.Dropout(rate=dropout)
        self.linear_2 = nnx.Linear(d_ff, d_model, rngs=nnx.Rngs(0))

    def __call__(self, x: jnp.ndarray):
        # simple feed forward network
        x = nnx.gelu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class MultiHeadAttentionBlock(nnx.Module):
    """
    Multi Head Attention Block
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        """

        Args:
            d_model: dimension of the model
            n_heads: number of heads
            dropout: dropout probability

        Returns:
            None
        """
        self.d_model = d_model
        self.n_heads = n_heads
        # self.dropout = dropout

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.w_q = nnx.Linear(d_model, d_model, rngs=nnx.Rngs(0))
        self.w_k = nnx.Linear(d_model, d_model, rngs=nnx.Rngs(0))
        self.w_v = nnx.Linear(d_model, d_model, rngs=nnx.Rngs(0))
        self.w_o = nnx.Linear(d_model, d_model, rngs=nnx.Rngs(0))
        self.dropout = nnx.Dropout(rate=dropout)

    @staticmethod
    def scaled_dot_product_attention(
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask,
        dropout: nnx.Dropout,
    ) -> jnp.ndarray:
        d_k = query.shape[-1]  # get dimension of last axis
        # (Q * K^T)/sqrt(d_k)
        attention_scores = jnp.matmul(query, key.transpose(-2, -1)) / jnp.sqrt(d_k)
        if mask:
            attention_scores = jnp.where(mask == 0, -1e10, attention_scores)
        # softmax(Q * K^T/sqrt(d_k))
        attention_scores = nnx.softmax(attention_scores, axis=-1)
        if dropout:
            attention_scores = dropout(attention_scores, dropout)
        # (Q * K^T)/sqrt(d_k) * V
        x = jnp.matmul(attention_scores, value)
        return x

    def __call__(self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, mask):
        """

        Args:
            q: query
            k: key
            v: value
            mask: mask

        Returns:
            None
        """
        # (seq_len, d_model) * (d_model, d_model) -> (seq_len, d_model)
        query: jnp.ndarray = self.w_q(q)
        key: jnp.ndarray = self.w_k(k)
        value: jnp.ndarray = self.w_v(v)

        # reshape using .view
        # (seq_len, d_model) -> (seq_len, n_heads, d_k) -> (n_heads, seq_len, d_k)
        # where dk = d_model // n_heads
        # (n, 512) -> (n, h, dk) -> (h, n, dk)
        # (3, 512) -> (3, 8, 64) -> (8, 3, 64)
        #
        # Sequence length n where each token is of dimension 512 ->
        # Sequence length n where each token is an array of 8 vectors of dimension 64 ->
        # Explaination: In a sequence the embeddings are split into 8 parts so that each head can focus on different parts of the embeddings
        # 8 Heads where each head contains a matrix of n vectors of dimension 64
        # keep the batch dimension the same and the sequence length the same
        # split the embeddings into 8 heads
        query = query.view(
            query.shape[0], query.shape[1], self.n_heads, self.d_k
        ).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(
            1, 2
        )
        value = value.view(
            value.shape[0], value.shape[1], self.n_heads, self.d_k
        ).transpose(1, 2)

        # apply scaled dot product attention to each head
        x = MultiHeadAttentionBlock.scaled_dot_product_attention(
            query, key, value, self.dropout
        )

        # reshape back to (seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)
        x = self.w_o(x)
        return x


class EncoderBlock(nnx.Module):
    def __init__(
        self,
        multi_head_attention_block: MultiHeadAttentionBlock,
        multi_layer_perceptron_block: MultiLayerPerceptron,
        dropout: float,
    ) -> None:
        """
        Args:
            multi_head_attention: multi head attention block
            feed_forward: feed forward block
            dropout: dropout probability

        Returns:
            None
        """
        # encoder block has one self attention block
        self.multi_head_attention_block = multi_head_attention_block
        # and one feed forward block
        self.multi_layer_perceptron_block = multi_layer_perceptron_block
        # Lastly there are two residual connections in the encoder block
        # that connect the multi head attention block and the feed forward block
        self.dropout = nnx.Dropout(rate=dropout)
        self.norm1 = LayerNorm()
        self.norm2 = LayerNorm()

    def __call__(self, x, src_mask):
        multi_head_attention_output = self.multi_head_attention_block(x, x, x, src_mask)

        x = self.dropout(self.norm1(multi_head_attention_output + x))

        multi_layer_perceptron_output = self.multi_layer_perceptron_block(x)

        output = self.dropout(self.norm2(multi_layer_perceptron_output + x))

        return output


class Encoder(nnx.Module):
    def __init__(self, blocks: nnx.List[EncoderBlock]) -> None:
        """
        Args:
            blocks: list of encoder blocks

        Returns:
            None
        """
        self.blocks = nnx.List(blocks)
        self.norm = LayerNorm()

    def __call__(self, x, mask):
        """
        Goes through all the encoder blocks

        Args:
            x: input
            mask: mask

        Returns:
            None
        """
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)


class ProjectionLayer(nnx.Module):
    """
    Projection layer to map the output of the decoder to the vocabulary. This gives us the logits

    Args:
        d_model: dimension of the model
        num_classes: the number of classes in our dataset

    Returns:
        None"""

    def __init__(self, d_model: int, num_classes: int) -> None:
        self.linear = nnx.Linear(d_model, num_classes, rngs=nnx.Rngs(0))

    def __call__(self, x):
        # (seq_len, d_model) -> (seq_len, vocab_size)
        return nnx.log_softmax(self.linear(x))


class VisionTransformer(nnx.Module):
    def __init__(
        self,
        encoder: Encoder,
        patch_embedding: PatchEmbedding,
        positional_encoding: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        self.encoder = encoder
        self.patch_embedding = patch_embedding
        self.positional_encoding = positional_encoding
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.patch_embedding(src)
        src = self.positional_encoding(src)
        return self.encoder(src, src_mask)
