import jax.numpy as jnp
from flax import linen as nn


class InputEmbeddings(nn.Module):
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
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)

    @nn.compact
    def __call__(self, x):
        # Get the embedding for each word in x
        # multiply by the square root of d_model for normalization and stability during training
        return self.embedding(x) * self.d_model**0.5


class PositionalEncoding(nn.Module):
    d_model: int
    seq_len: int
    dropout_rate: float

    def setup(self):
        """
        Args:
            d_model: embedding dimension
            seq_len: maximum input sequence length
            dropout: dropout rate (used during training)
        """

        self.dropout = nn.Dropout(rate=self.dropout_rate)

        self.pe = self.param(
            "positional_encoding",
            nn.initializers.zeros,
            (1, self.seq_len, self.d_model),
        )

    def __call__(self, x: jnp.ndarray, is_training: bool):
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)
            training: whether in training mode for dropout
        """
        seq_len = x.shape[1]  # actual runtime sequence length
        pe = self.pe[:, :seq_len, :]  # slice to match

        x = x + pe
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
    def __call__(self, x: jnp.ndarray):
        # compute mean and std for each patch in the sequence
        # (batch, seq_len, d_model)
        # axis=-1 means along the last dimension which is d_model
        # (batch_size, seq_len, 1) this holds mean of each token in the sequence
        mean = jnp.mean(x, axis=-1, keepdims=True)
        # (batch_size, seq_len, 1) var of each token in the sequence
        var = jnp.var(x, axis=-1, keepdims=True)
        # all elements in x are normalized by mean and std
        return (self.alpha * (x - mean) / jnp.sqrt(var + self.eps)) + self.bias


class FeedForwardBlock(nn.Module):
    d_model: int
    d_ff: int
    dropout_rate: float

    def setup(self) -> None:
        """
        Args:
            d_model: dimension of the model
            d_ff: dimension of the feed forward network
            dropout: dropout probability

        Returns:
            None
        """

        self.linear_1 = nn.Dense(features=self.d_ff, dtype=jnp.bfloat16)
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.linear_2 = nn.Dense(features=self.d_model, dtype=jnp.bfloat16)

    @nn.compact
    def __call__(self, x: jnp.ndarray, is_training: bool):
        # simple feed forward network
        # (seq_len, d_model) --> (dff, d_model) --> (seq_len, d_model)
        x = nn.leaky_relu(self.linear_1(x))
        x = self.dropout(x, deterministic=not is_training)
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

        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = self.d_model // self.n_heads  # size of each head
        self.w_q = nn.Dense(features=self.d_model, dtype=jnp.bfloat16)
        self.w_k = nn.Dense(features=self.d_model, dtype=jnp.bfloat16)
        self.w_v = nn.Dense(features=self.d_model, dtype=jnp.bfloat16)
        self.w_o = nn.Dense(features=self.d_model, dtype=jnp.bfloat16)
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    @staticmethod
    def flash_attention(
        query: jnp.ndarray,  # (B, H, L_t, d_k)
        key: jnp.ndarray,  # (B, H, L_s, d_k)
        value: jnp.ndarray,  # (B, H, L_s, d_k)
        mask: jnp.ndarray,  # None or (B, 1, 1, L_s) or broadcastable
        is_training: bool = False,
        dropout: nn.Dropout = None,
        q_block_size: int = 64,
        k_block_size: int = 64,
    ) -> jnp.ndarray:
        """
        Streaming (tiled) exact attention — FlashAttention style.

        Assumes inputs are shaped (B, H, L, d_k). Returns (B, H, L_t, d_k).
        All internal math is done in float32 for stability, result cast back.
        """
        if k_block_size is None:
            k_block_size = q_block_size

        B, H, L_target, d_k = query.shape  # batch_size, n_heads, seq_len, d_k
        _, _, L_src, _ = key.shape  # batch_size, n_heads, seq_len, d_k
        scale = 1.0 / jnp.sqrt(d_k)  # scale factor

        # work in float32 for stability (inputs may be bfloat16)
        query = query.astype(jnp.float32)
        key = key.astype(jnp.float32)
        value = value.astype(jnp.float32)

        # prepare mask broadcast rules: mask should index keys
        # Accept mask shapes: (B, L_s), (B, 1, 1, L_s), or None
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, None, None, :]  # (B,1,1,L_s)
            elif mask.ndim == 3:
                mask = mask[:, :, None, :]  # (B,1,1,L_s) expected
            # mask now (B, 1, 1, L_s) or broadcastable

        # We'll produce output in blocks of queries
        out_blocks = []

        # iterate over query blocks
        for q_start in range(0, L_target, q_block_size):
            q_end = min(q_start + q_block_size, L_target)
            # get query block
            q_block = query[:, :, q_start:q_end, :]  # (B, H, Qb, d_k)

            # initialize accumulators for this q_block
            # m = running max of scores for each (B,H,Qb)
            running_max = jnp.full((B, H, q_end - q_start), -jnp.inf, dtype=jnp.float32)
            # l = running sum of exp(scores - m) for normalization, shape (B,H,Qb)
            running_sum = jnp.zeros((B, H, q_end - q_start), dtype=jnp.float32)
            # acc = running weighted sum of V, shape (B,H,Qb,d_k)
            running_acc = jnp.zeros((B, H, q_end - q_start, d_k), dtype=jnp.float32)

            # iterate over key/value blocks
            for k_start in range(0, L_src, k_block_size):
                k_end = min(k_start + k_block_size, L_src)
                # get key/value blocks
                k_block = key[:, :, k_start:k_end, :]  # (B,H,Kb,d_k)
                v_block = value[:, :, k_start:k_end, :]  # (B,H,Kb,d_k)
                q_pos = jnp.arange(q_start, q_end)[:, None]  # (Qb,1)
                k_pos = jnp.arange(k_start, k_end)[None, :]  # (1,Kb)
                causal_block = q_pos >= k_pos  # (Qb,Kb)

                # compute scores for this tile: (B,H,Qb,Kb)
                # einsum is clear: q @ k^T
                scores = jnp.matmul(q_block, k_block.swapaxes(-2, -1)) * scale

                # apply mask for the keys in this block (if provided)
                if mask is not None:
                    # ensure mask is boolean
                    padding_block = mask[:, :, :, k_start:k_end].astype(
                        bool
                    )  # <- cast to bool
                    mask_block = causal_block[None, None, :, :] & padding_block
                else:
                    mask_block = causal_block[None, None, :, :]

                scores = jnp.where(mask_block, scores, -jnp.inf)

                # per-query-row max in this new tile
                m_block = jnp.max(scores, axis=-1)  # (B,H,Qb)

                # new running max
                m_new = jnp.maximum(running_max, m_block)  # (B,H,Qb)

                # compute exp(S - m_new[...,None]) safely
                exp_scores = jnp.exp(scores - m_new[..., None])  # (B,H,Qb,Kb)

                # sum over keys in block
                exp_sum = jnp.sum(exp_scores, axis=-1)  # (B,H,Qb)

                # update l: l_new = exp(m - m_new)*l + exp_sum
                # factor = exp(m - m_new) (<= 1)
                factor = jnp.exp(running_max - m_new)
                running_sum = factor * running_sum + exp_sum  # (B,H,Qb)

                # weighted value: sum_k exp(S - m_new) * V_block
                # compute (B,H,Qb,d_k) <- sum_k exp_S * V_block
                # first expand exp_S to (B,H,Qb,Kb,1) and multiply by V_block (B,H,Kb,d_k)
                weighted_v = jnp.einsum("bhqk,bhkd->bhqd", exp_scores, v_block)
                # weighted_v = jnp.matmul(exp_scores, v_block)

                # update acc similarly: acc_new = factor[...,None]*acc + weighted_v
                running_acc = factor[..., None] * running_acc + weighted_v

                # set m <- m_new for next iteration
                running_max = m_new

            # After all key blocks processed: normalize acc / l[...,None]
            out_block = running_acc / (
                running_sum[..., None] + -jnp.inf
            )  # (B,H,Qb,d_k)
            out_blocks.append(out_block)

        # concatenate along query length axis
        out = jnp.concatenate(out_blocks, axis=2)  # (B,H,L_t,d_k)
        return out.astype(query.dtype)

    @nn.compact
    def __call__(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        mask: jnp.ndarray,
        is_training: bool,
    ) -> jnp.ndarray:
        """

        Args:
            q: query
            k: key
            v: value

        Returns:
            jnp.ndarray
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

        # with the encoder these will be the same
        B, L_target, _ = query.shape
        _, L_src, _ = key.shape

        # (b, seq_len, d_model) -> (b, n_heads, seq_len, d_k)
        # split into n_heads then order axes as (b, h, seq_len, d_k)
        query = query.reshape(B, L_target, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        key = key.reshape(B, L_src, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        value = value.reshape(B, L_src, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # apply scaled dot product attention to each head
        x = MultiHeadAttentionBlock.flash_attention(
            query=query,
            key=key,
            value=value,
            dropout=self.dropout,
            is_training=is_training,
            mask=mask,
        )

        # reshape back to (seq_len, d_model)
        # order axis as (b, seq_len, h, d_k) then reshape (b, seq_len, d_model)
        x = jnp.transpose(x, (0, 2, 1, 3)).reshape(B, L_target, self.d_model)
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
        self.feed_forward_block = FeedForwardBlock(
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
    def __call__(self, x: jnp.ndarray, src_mask: jnp.ndarray, is_training: bool):
        # attention block output
        multi_head_attention_output = self.multi_head_attention_block(
            q=x, k=x, v=x, mask=src_mask, is_training=is_training
        )

        # add and norm output
        x = self.dropout(
            self.norm1(multi_head_attention_output + x), deterministic=not is_training
        )

        # pass in new x into feed forward and get output
        feed_forward_output = self.feed_forward_block(x, is_training=is_training)

        # add and norm ff output
        output = self.dropout(
            self.norm2(feed_forward_output + x), deterministic=not is_training
        )

        return output


class Encoder(nn.Module):
    encoder_blocks: list[EncoderBlock]
    d_model: int

    def setup(self) -> None:
        """
        Args:
            blocks: list of encoder blocks

        Returns:
            None
        """
        self.blocks: list[EncoderBlock] = self.encoder_blocks
        self.norm: LayerNorm = LayerNorm(d_model=self.d_model)

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray, is_training: bool):
        for block in self.blocks:
            x = block(x=x, src_mask=mask, is_training=is_training)
        return self.norm(x)


class DecoderBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float

    def setup(
        self,
    ) -> None:
        self.masked_multi_head_attention_block = MultiHeadAttentionBlock(
            d_model=self.d_model, n_heads=self.n_heads, dropout_rate=self.dropout_rate
        )
        self.cross_attention_block = MultiHeadAttentionBlock(
            d_model=self.d_model, n_heads=self.n_heads, dropout_rate=self.dropout_rate
        )
        self.feed_forward_block = FeedForwardBlock(
            d_model=self.d_model, d_ff=self.d_ff, dropout_rate=self.dropout_rate
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm1 = LayerNorm(d_model=self.d_model)
        self.norm2 = LayerNorm(d_model=self.d_model)
        self.norm3 = LayerNorm(d_model=self.d_model)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        encoder_output: jnp.ndarray,
        src_mask: jnp.ndarray,
        target_mask: jnp.ndarray,
        is_training: bool,
    ):
        # masked multi head attention block output
        masked_multi_head_attention_output = self.masked_multi_head_attention_block(
            q=x, k=x, v=x, mask=target_mask, is_training=is_training
        )

        # add and norm the masked multi head attention
        x = self.dropout(
            self.norm1(masked_multi_head_attention_output + x),
            deterministic=not is_training,
        )

        # cross attention
        cross_attention_output = self.cross_attention_block(
            q=x,
            k=encoder_output,
            v=encoder_output,
            mask=src_mask,
            is_training=is_training,
        )

        # add and norm the cross attention
        x = self.dropout(
            self.norm2(cross_attention_output + x), deterministic=not is_training
        )

        # feed forward
        feed_forward_output = self.feed_forward_block(x, is_training=is_training)

        # final add and norm
        output = self.dropout(
            self.norm3(feed_forward_output + x), deterministic=not is_training
        )

        return output


class Decoder(nn.Module):
    decoder_blocks: list[DecoderBlock]
    d_model: int

    def setup(self) -> None:
        """
        Args:
            blocks: list of decoder blocks

        Returns:
            None
        """
        self.blocks: list[DecoderBlock] = self.decoder_blocks
        self.norm: LayerNorm = LayerNorm(d_model=self.d_model)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        encoder_output: jnp.ndarray,
        src_mask: jnp.ndarray,
        target_mask: jnp.ndarray,
        is_training: bool,
    ):
        for block in self.blocks:
            x = block(
                x=x,
                encoder_output=encoder_output,
                src_mask=src_mask,
                target_mask=target_mask,
                is_training=is_training,
            )
        return self.norm(x)


class ProjectionLayer(nn.Module):
    """
    Projection layer to map the output of the decoder to the vocabulary. This gives us the logits

    Args:
        d_model: dimension of the model
        vocab_size: size of the vocabulary

    Returns:
        None
    """

    vocab_size: int

    def setup(self) -> None:
        self.linear = nn.Dense(features=self.vocab_size, dtype=jnp.bfloat16)

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, vocab_size)
        return self.linear(x)


class Transformer(nn.Module):
    d_model: int
    N: int
    n_heads: int
    d_ff: int
    dropout: float
    seq_len: int
    src_vocab_size: int
    target_vocab_size: int

    def setup(
        self,
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
            d_model=self.d_model, vocab_size=self.src_vocab_size
        )
        self.src_pe = PositionalEncoding(
            d_model=self.d_model, seq_len=self.seq_len, dropout_rate=self.dropout
        )

        self.target_embeddings = InputEmbeddings(
            d_model=self.d_model, vocab_size=self.target_vocab_size
        )
        self.target_pe = PositionalEncoding(
            d_model=self.d_model, seq_len=self.seq_len, dropout_rate=self.dropout
        )

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

        self.decoder = Decoder(
            decoder_blocks=[
                DecoderBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_ff=self.d_ff,
                    dropout_rate=self.dropout,
                )
                for _ in range(self.N)
            ],
            d_model=self.d_model,
        )

        self.projection = ProjectionLayer(vocab_size=self.target_vocab_size)

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

        return output
