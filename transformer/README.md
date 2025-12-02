I am trying to train a transformer model, This is my init "import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from transformer.model import Transformer
from utils.config import Config


def init_train_state(config: Config, vocab_size: int) -> train_state.TrainState:
    """
    Initialize the train state
    Args:
        config: Config

    Returns:
        train_state.TrainState
    """
    model: Transformer = Transformer(
        d_model=config.D_MODEL,
        N=config.N,
        n_heads=config.H,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        seq_len=config.SEQ_LEN,
        src_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
    )

    rng: jax.random.PRNGKey = jax.random.PRNGKey(0)

    # create dummy inputs
    dummy_src_input = jnp.zeros(
        (config.BATCH_SIZE, config.SEQ_LEN),
        dtype=jnp.int32,
    )

    dummy_src_mask = jnp.ones(
        (config.BATCH_SIZE, 1, 1, config.SEQ_LEN), dtype=jnp.float32
    )

    dummy_target_input = jnp.zeros(
        (config.BATCH_SIZE, config.SEQ_LEN - 1),
        dtype=jnp.int32,
    )

    dummy_target_mask = jnp.ones(
        (config.BATCH_SIZE, 1, 1, config.SEQ_LEN), dtype=jnp.float32
    )
    # Initialize with dummy inputs
    variables = model.init(
        rng,
        src=dummy_src_input,
        src_mask=dummy_src_mask,
        target=dummy_target_input,
        target_mask=dummy_target_mask,
        is_training=True,
    )

    params = variables["params"]" and this is my multi head attention that implements flash attention "class MultiHeadAttentionBlock(nn.Module):
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

                # compute scores for this tile: (B,H,Qb,Kb)
                # einsum is clear: q @ k^T
                scores = jnp.matmul(q_block, k_block.swapaxes(-2, -1)) * scale
                # apply mask for the keys in this block (if provided)
                if mask is not None:
                    # mask slice: (B,1,1,Kb) or broadcastable
                    mask_block = mask[:, :, :, k_start:k_end]  # (B,1,1,Kb)
                    # mask == 0 should be blocked; set to -inf
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
" and this is my data loader "    def create_src_mask(self, src):
        """
        Create source mask for encoder. Fill with 1 for tokens and 0 for padding
        Changes shape from (b, seq_len) to (b, 1, 1, seq_len)
        Args:
            src: source sequence
        Returns:
            source mask
        """
        return (src != 0).astype(jnp.float32)[
            :, None, None, :
        ]  # 1 for tokens, 0 for padding

    def create_target_mask(self, target):
        """
        Create target mask for encoder. Fill padding mask with 1 for tokens and 0 for padding
        Changes shape from (b, seq_len) to (b, 1, 1, seq_len)
        We then create a lower triangular matrix where future tokens are masked.
        Multiply this with the source mask to ignore padding tokens and we get final mask.
        Args:
            target: target sequence
        Returns:
            source mask
        """
        padding_mask = (target != 0).astype(jnp.float32)[:, None, None, :]  # (B,1,1,L)
        # create causal mask (1,L,L) (lower triangular matrix)
        causal_mask = jnp.tril(
            jnp.ones((self.seq_len, self.seq_len), dtype=jnp.float32)
        )[None, None, :, :]
        # create final mask where future tokens are masked and padding tokens are ignored
        return padding_mask * causal_mask  # (B,1,L,L)

    def __iter__(self, rng=None):
        return self._prefetch_to_device(self._batch_generator(rng))

    def _batch_generator(self, rng):
        N = self.src.shape[0]
        indices = jnp.arange(N)

        # Only shuffle if RNG provided by trainer
        if self.shuffle:
            if rng is None:
                raise ValueError("Shuffle=True but no RNG was provided to DataLoader")
            indices = jax.random.permutation(rng, indices)

        for start in range(0, N, self.batch_size):
            end = start + self.batch_size
            batch_idx = indices[start:end]
            if len(batch_idx) < self.batch_size:
                break

            batch_src = jnp.array(self.src[batch_idx])
            batch_target = jnp.array(self.target[batch_idx])

            src_mask = self.create_src_mask(batch_src)
            target_attention_mask = self.create_target_mask(batch_target)
            token_mask = (batch_target[:, 1:] != 0).astype(jnp.float32)
            # target_key_mask = (batch_target[:, 1:] != 0).astype(jnp.float32)[
            #     :, None, None, :
            # ]

            yield {
                "src": batch_src,
                "src_mask": src_mask,
                "target_input": batch_target[:, :-1],
                # "target_key_mask": target_key_mask,
                "target_output": batch_target[:, 1:],
                "target_mask": target_attention_mask[
                    :, :, : self.seq_len, : self.seq_len
                ],
                "token_mask": token_mask,
            }" I am having trouble correctly using the masks. The current loader pads and src and target masks and also creates a lower triangular matrix for the target. Help me implement masks corrcetly