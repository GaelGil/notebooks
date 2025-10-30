import jax.numpy as jnp
from flax import linen as nn


class PatchEmbedding(nn.Module):
    """
    Projects the image into patches

    """

    patch_size: int
    d_model: int

    def setup(
        self,
    ) -> None:
        """
        Args:
            None

        Returns:
            None
        """

        # project the image into patches of size patch_size
        # each conv/patch will learn a representation of the image
        self.projection = nn.Conv(
            features=self.d_model,
            kernel_size=self.patch_size,
            strides=self.patch_size,
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
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)

        return x


class PositionalEncoding(nn.Module):
    d_model: int
    num_patches: int
    dropout_rate: float = 0.1

    def setup(self):
        """
        Args:
            d_model: embedding dimension
            seq_len: maximum input sequence length
            dropout: dropout rate (used during training)
        """
        self.dropout = nn.Dropout(rate=self)

        # create cls token with shape (1, 1, d_model)
        self.cls_token = self.param(
            "cls_token", nn.initializers.zeros, (1, 1, self.d_model)
        )
        # create positional encoding with shape (1, num_patches + 1, d_model)
        self.pe = self.param(
            "positional_encoding",
            nn.initializers.zeros,
            (1, self.num_patches + 1, self.d_model),
        )

    def __call__(self, x: jnp.ndarray, training: bool = False):
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)
            training: whether in training mode for dropout
        """
        # get batch size
        B = x.shape[0]

        # duplicate cls token B times so each batch has its own cls token
        cls = jnp.tile(self.cls_token, (B, 1, 1))
        # concatenate cls token to sequence (batch_size, seq_len + 1, d_model)
        x = jnp.concatenate([cls, x], axis=1)

        # add positional encoding (batch_size, seq_len + 1, d_model)
        x = x + self.pe.value

        return self.dropout(x, deterministic=not training)


class LayerNorm(nn.Module):
    eps: float = 1e-6  # helps avoid division by zero

    def setup(self) -> None:
        """
        Args:
            eps: epsilon for numerical stability

        Returns:
            None
        """

        self.alpha = self.param("alpha", nn.initializers.ones, (1))
        self.bias = self.param("bias", nn.initializers.zeros, (1))

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


class MultiLayerPerceptron(nn.Module):
    d_model: int
    d_ff: int
    dropout_rate: float

    def setup(
        self,
    ) -> None:
        """
        Args:
            d_model: dimension of the model
            d_ff: dimension of the feed forward network
            dropout: dropout probability

        Returns:
            None
        """

        # self.dropout = dropout
        self.linear_1 = nn.Dense(features=self.d_ff)
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.linear_2 = nn.Dense(features=self.d_model)

    def __call__(self, x: jnp.ndarray):
        # simple feed forward network
        x = nn.gelu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class MultiHeadAttentionBlock(nn.Module):
    """
    Multi Head Attention Block
    """

    d_model: int
    n_heads: int
    dropout_rate: float

    def setup(self) -> None:
        """

        Args:
            d_model: dimension of the model
            n_heads: number of heads
            dropout: dropout probability

        Returns:
            None
        """

        # self.dropout = dropout

        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = self.d_model // self.n_heads
        self.w_q = nn.Dense(features=self.d_model)
        self.w_k = nn.Dense(features=self.d_model)
        self.w_v = nn.Dense(features=self.d_model)
        self.w_o = nn.Dense(features=self.d_model)
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    @staticmethod
    def scaled_dot_product_attention(
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask,
        dropout: nn.Dropout,
    ) -> jnp.ndarray:
        """
        For each head, compute  softmax(Q * K^T/sqrt(d_k)) * V
        Args:
            query: query
            key: key
            value: value
            mask: mask
            dropout: dropout

        Returns:
            None
        """
        d_k = query.shape[-1]  # get dimension of last axis
        # (Q * K^T)/sqrt(d_k)
        attention_scores = jnp.matmul(query, key.transpose(-2, -1)) / jnp.sqrt(d_k)
        if mask:
            attention_scores = jnp.where(mask == 0, -1e10, attention_scores)
        # softmax(Q * K^T/sqrt(d_k))
        attention_scores = nn.softmax(attention_scores, axis=-1)
        if dropout:
            attention_scores = dropout(attention_scores, dropout)
        # softmax((Q * K^T)/sqrt(d_k)) * V
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


class EncoderBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float

    def setup(self) -> None:
        """
        Args:
            multi_head_attention: multi head attention block
            feed_forward: feed forward block
            dropout: dropout probability

        Returns:
            None
        """
        # encoder block has one self attention block
        self.multi_head_attention_block = MultiHeadAttentionBlock(
            d_model=self.d_model, n_heads=self.n_heads, dropout_rate=self.dropout_rate
        )
        # and one feed forward block
        self.multi_layer_perceptron_block = MultiLayerPerceptron(
            d_model=self.d_model, d_ff=self.d_ff, dropout_rate=self.dropout_rate
        )
        # Lastly there are two residual connections in the encoder block
        # that connect the multi head attention block and the feed forward block
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.norm1 = LayerNorm()
        self.norm2 = LayerNorm()

    def __call__(self, x, src_mask):
        multi_head_attention_output = self.multi_head_attention_block(x, x, x, src_mask)

        x = self.dropout(self.norm1(multi_head_attention_output + x))

        multi_layer_perceptron_output = self.multi_layer_perceptron_block(x)

        output = self.dropout(self.norm2(multi_layer_perceptron_output + x))

        return output


class Encoder(nn.Module):
    encoder_blocks: nn.Sequential

    def setup(self) -> None:
        """
        Args:
            blocks: list of encoder blocks

        Returns:
            None
        """
        self.blocks: nn.Sequential = self.encoder_blocks
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


class ProjectionLayer(nn.Module):
    """
    Projection layer to map the output of the decoder to the vocabulary. This gives us the logits

    """

    num_classes: int

    def setup(self, num_classes: int) -> None:
        """
        Args:
            d_model: dimension of the model
            num_classes: the number of classes in our dataset

        Returns:
            None
        """
        self.linear = nn.Dense(features=num_classes)

    def __call__(self, x):
        # (seq_len, d_model) -> (seq_len, vocab_size)
        return nn.log_softmax(self.linear(x))


class VisionTransformer(nn.Module):
    N: int
    n_heads: int
    dropout: float
    img_size: int
    patch_size: int
    in_channels: int
    d_model: int
    d_ff: int
    num_classes: int

    def setup(
        self,
    ) -> None:
        """
        Args:
            N: number of encoder blocks
            n_heads: number of heads
            dropout: dropout probability
            img_size: image size
            patch_size: patch size
            in_channels: number of channels
            d_model: dimension of the model
            num_classes: the number of classes in our dataset

        Returns:
            None
        """

        self.patch_embedding = PatchEmbedding(
            patch_size=self.patch_size,
            d_model=self.d_model,
        )
        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            num_patches=(self.img_size // self.patch_size) ** 2,
            dropout_rate=self.dropout,
        )
        self.projection_layer = ProjectionLayer(num_classes=self.num_classes)

        self.encoder = Encoder(
            nn.Sequential(
                layers=[
                    EncoderBlock(
                        d_model=self.d_model,
                        n_heads=self.n_heads,
                        d_ff=self.d_ff,
                        dropout_rate=self.dropout,
                    )
                    for _ in range(self.N)
                ]
            )
        )

    def __call__(self, x, src_mask):
        x = self.patch_embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x, src_mask)
        x = self.projection_layer(x)
        return x
