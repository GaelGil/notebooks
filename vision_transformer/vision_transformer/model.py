import jax.numpy as jnp
from flax import linen as nn


class PatchEmbedding(nn.Module):
    """
    Projects the image into patches

    Attributes:
        patch_size: size of the patch
        d_model: dimension of the model

    """

    patch_size: int
    d_model: int

    def setup(
        self,
    ) -> None:
        """
        Create the patch embedding layer. This is a convolutional that will
        project the image into n patches of size patch_size

        Args:
            None
        Returns:
            None
        """

        # project the image into patches of size patch_size
        # each conv/patch will learn a representation of the image
        self.projection = nn.Conv(
            features=self.d_model,
            kernel_size=(self.patch_size, self.patch_size),
            strides=self.patch_size,
            padding="VALID",
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """
        Projects the image into patches

        Args:
            x: image

        Returns:
            x projected into patches
        """
        x = self.projection(x)
        # reshape x into (batch_size, num_patches, d_model)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        # returns (batch_size, num_patches, d_model)
        return x


class PositionalEncoding(nn.Module):
    """
    Attributes:
        d_model: dimension of the model
        num_patches: number of patches
        training: whether in training mode
        dropout_rate: dropout rate
    """

    d_model: int
    num_patches: int
    dropout_rate: float = 0.1

    def setup(self):
        """
        Create the positional encoding layer. This is a learnable parameter that we add to the sequence.
        We also create a cls token that we append to the sequence.
        Args:
            None

        Returns:
            None
        """
        self.dropout = nn.Dropout(rate=self.dropout_rate)

        # create cls token with shape (1, 1, d_model)
        # this token is a embedding and will be added to the num patches
        self.cls_token = self.param(
            "cls_token", nn.initializers.zeros, (1, 1, self.d_model)
        )
        # create positional encoding matrix with shape (1, num_patches + 1, d_model)
        # Each row in the positional encoding matrix is a vector that is added to a
        # corresponding patch
        self.pe = self.param(
            "positional_encoding",
            nn.initializers.zeros,
            (1, self.num_patches + 1, self.d_model),
        )
        nn.LayerNorm(d_model=self.d_model)

    @nn.compact
    def __call__(self, x: jnp.ndarray, is_training: bool = True):
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

        # add positional encoding to sequence
        x = x + self.pe

        # apply dropout
        # returns (batch_size, num_patches + 1, d_model)
        return self.dropout(x, deterministic=not is_training)


class LayerNorm(nn.Module):
    d_model: int
    eps: float = 1e-6  # helps avoid division by zero

    def setup(self) -> None:
        """Set up layer norm
        Args:
            None

        Returns:
            None
        """
        # create alpha and bias of shape (d_model)
        # alpha and bias are learnable parameters
        # alpha and bias are applied to each patch
        self.alpha = self.param("alpha", nn.initializers.ones, (self.d_model))
        self.bias = self.param("bias", nn.initializers.zeros, (self.d_model))

    @nn.compact
    def __call__(self, x):
        # compute mean and std for each patch in the sequence
        # (batch, num_patches, d_model)
        # axis=-1 means along the last dimension which is d_model
        # (batch_size, num_patches, 1) this holds mean of each feature
        mean = jnp.mean(x, axis=-1, keepdims=True)
        # (batch_size, num_patches, 1) holds std of each feature
        var = jnp.var(x, axis=-1, keepdims=True)
        # all elements in x are normalized by mean and std
        return (self.alpha * (x - mean) / jnp.sqrt(var + self.eps)) + self.bias


class MultiLayerPerceptron(nn.Module):
    """
    Attributes:
        d_model: dimension of the model
        d_ff: dimension of the feed forward network
        dropout_rate: dropout rate
        training: whether in training mode
    """

    d_model: int
    d_ff: int
    dropout_rate: float

    def setup(
        self,
    ) -> None:
        """
        Create a feed forward network. With two linear layers.
        It should follow (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        Args:
            None

        Returns:
            None
        """
        self.linear_1 = nn.Dense(features=self.d_ff)
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.linear_2 = nn.Dense(features=self.d_model)

    @nn.compact
    def __call__(self, x: jnp.ndarray, is_training: bool):
        # simple feed forward network
        x = nn.gelu(self.linear_1(x))
        x = self.dropout(x, deterministic=not is_training)
        x = self.linear_2(x)
        return x


class MultiHeadAttentionBlock(nn.Module):
    """
    Multi Head Attention Block

    Attributes:
        d_model: dimension of the model
        n_heads: number of heads
        dropout_rate: dropout rate
        training: whether in training mode
    """

    d_model: int
    n_heads: int
    dropout_rate: float

    def setup(self) -> None:
        """
        Set up multi head attention block. Each query, key, value gets its own linear layer. We will multiply
        w_q by q -> (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model) then split into
        n_heads -> (batch_size, seq_len, n_heads, d_k). Which means we have a sequence length each with 8 heads of size d_k
        We will then compute scaled dot product attention for each head.

        Args:
            None

        Returns:
            None
        """

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
        dropout: nn.Dropout,
        training: bool,
    ) -> jnp.ndarray:
        """
        For each head, compute  softmax(Q * K^T/sqrt(d_k)) * V
        Args:
            query: query
            key: key
            value: value
            dropout: dropout

        Returns:
            None
        """
        d_k = query.shape[-1]  # get dimension of last axis
        # (Q * K^T)/sqrt(d_k)
        attention_scores = jnp.matmul(query, key.swapaxes(-2, -1)) / jnp.sqrt(d_k)
        # softmax(Q * K^T/sqrt(d_k))
        attention_scores = nn.softmax(attention_scores, axis=-1)
        if dropout:
            attention_scores = dropout(attention_scores, deterministic=not training)
        # softmax((Q * K^T)/sqrt(d_k)) * V
        x = jnp.matmul(attention_scores, value)
        return x

    @nn.compact
    def __call__(
        self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, is_training: bool
    ):
        """

        Args:
            q: query
            k: key
            v: value

        Returns:
            None
        """
        # (seq_len, d_model) * (d_model, d_model) -> (seq_len, d_model)
        query: jnp.ndarray = self.w_q(q)
        key: jnp.ndarray = self.w_k(k)
        value: jnp.ndarray = self.w_v(v)

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
        query = query.reshape(
            query.shape[0], query.shape[1], self.n_heads, self.d_k
        ).transpose(0, 2, 1, 3)
        key = key.reshape(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(
            0, 2, 1, 3
        )
        value = value.reshape(
            value.shape[0], value.shape[1], self.n_heads, self.d_k
        ).transpose(0, 2, 1, 3)

        # apply scaled dot product attention to each head
        x = MultiHeadAttentionBlock.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            dropout=self.dropout,
            training=is_training,
        )

        # reshape back to (seq_len, d_model)
        # x = x.transpose(1, 2).contiguous().reshape(x.shape[0], -1, self.d_model)
        x = jnp.transpose(x, (0, 2, 1, 3)).reshape(x.shape[0], -1, self.d_model)
        x = self.w_o(x)
        return x


class EncoderBlock(nn.Module):
    """
    Atttributes:
        d_model: dimension of model
        n_heads: number of heads
        d_ff: dimension of feed forward network
        dropout_rate: dropout rate
        training: whether in training mode
    """

    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float

    def setup(self) -> None:
        """
        Set up encoder block
        Each encoder block has one multi head attention block and one feed forward block.
        There is also the residual connections of which it has two.
        Args:
            None

        Returns:
            None
        """
        # encoder block has one self attention block
        self.multi_head_attention_block = MultiHeadAttentionBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate,
        )
        # and one feed forward block
        self.multi_layer_perceptron_block = MultiLayerPerceptron(
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout_rate=self.dropout_rate,
        )
        # Lastly there are two residual connections in the encoder block
        # that connect the multi head attention block and the feed forward block
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.norm1 = LayerNorm(d_model=self.d_model)
        self.norm2 = LayerNorm(d_model=self.d_model)

    @nn.compact
    def __call__(self, x, is_training: bool):
        multi_head_attention_output = self.multi_head_attention_block(
            q=x, k=x, v=x, is_training=is_training
        )

        x = self.dropout(
            self.norm1(multi_head_attention_output + x), deterministic=not is_training
        )

        multi_layer_perceptron_output = self.multi_layer_perceptron_block(
            x=x, is_training=is_training
        )

        output = self.dropout(
            self.norm2(multi_layer_perceptron_output + x), deterministic=not is_training
        )

        return output


class Encoder(nn.Module):
    """
    Attributes:
        encoder_blocks: A sequence of encoder blocks
        d_model: dimension of model
    """

    encoder_blocks: list[EncoderBlock]
    d_model: int

    def setup(self) -> None:
        """
        Set up encoder with a sequence of encoder blocks
        Args:
            None

        Returns:
            None
        """
        self.blocks = self.encoder_blocks
        self.norm = LayerNorm(d_model=self.d_model)

    @nn.compact
    def __call__(self, x, *, is_training: bool):
        """
        Goes through all the encoder blocks

        Args:
            x: input

        Returns:
            None
        """
        for block in self.blocks:
            x = block(x, is_training=is_training)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    """
    Projection layer to map the output of the decoder to the number of classes. This gives us the logits

    Attributes:
        num_classes: the number of classes in our dataset
    """

    num_classes: int

    def setup(self) -> None:
        """
        Set up projection layer. Should map the output of the encoder to the number of classes in our dataset
        Args:
            None

        Returns:
            None
        """
        self.linear = nn.Dense(features=self.num_classes)

    @nn.compact
    def __call__(self, x):
        cls_token = x[:, 0]  # shape: (batch_size, d_model) get the cls token
        logits = self.linear(
            cls_token
        )  # shape: (batch_size, num_classes) get the logits
        return logits  # get logits/raw output


class VisionTransformer(nn.Module):
    """
    Attributes:
        N: number of encoder blocks
        n_heads: number of heads
        dropout: dropout probability
        img_size: image size
        patch_size: patch size
        in_channels: number of channels
        d_model: dimension of the model
        num_classes: the number of classes in our dataset

    """

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
        Set up the vision transformer. With a patch embedding, positional encoding, projection layer and encoder
        Args:
            None


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
            encoder_blocks=[
                EncoderBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_ff=self.d_ff,
                    dropout_rate=self.dropout,
                )
                for _ in range(self.N)
            ],
            d_model=self.d_model,
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray, is_training: bool):
        x = self.patch_embedding(x=x)
        x = self.positional_encoding(x=x, is_training=is_training)
        x = self.encoder(x=x, is_training=is_training)
        x = self.projection_layer(x)
        return x
