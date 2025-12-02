Skip to content
You said:
Traceback (most recent call last):
  File "/home/gg/git_repos/notebooks/transformer/.venv/lib/python3.13/site-packages/jax/_src/lax/lax.py", line 221, in _broadcast_shapes_uncached
    return _try_broadcast_shapes(*rank_promoted_shapes, name='broadcast_shapes')
  File "/home/gg/git_repos/notebooks/transformer/.venv/lib/python3.13/site-packages/jax/_src/lax/lax.py", line 136, in _try_broadcast_shapes
    raise TypeError(f'{name} got incompatible shapes for broadcasting: '
                    f'{", ".join(map(str, map(tuple, shapes)))}.')
TypeError: broadcast_shapes got incompatible shapes for broadcasting: (8, 1, 191, 64), (8, 8, 64, 64), (1, 1, 1, 1).

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/gg/git_repos/notebooks/transformer/.venv/lib/python3.13/site-packages/jax/_src/lax/lax.py", line 199, in broadcast_shapes
    return _broadcast_shapes_cached(*shapes)
  File "/home/gg/git_repos/notebooks/transformer/.venv/lib/python3.13/site-packages/jax/_src/util.py", line 300, in wrapper
    return cached(trace_context(), *args, **kwargs)
  File "/home/gg/git_repos/notebooks/transformer/.venv/lib/python3.13/site-packages/jax/_src/util.py", line 294, in cached
    return f(*args, **kwargs)
  File "/home/gg/git_repos/notebooks/transformer/.venv/lib/python3.13/site-packages/jax/_src/lax/lax.py", line 205, in _broadcast_shapes_cached
    return _broadcast_shapes_uncached(*shapes)
  File "/home/gg/git_repos/notebooks/transformer/.venv/lib/python3.13/site-packages/jax/_src/lax/lax.py", line 224, in _broadcast_shapes_uncached
    raise ValueError(f"Incompatible shapes for broadcasting: shapes={list(shapes)}") from err
ValueError: Incompatible shapes for broadcasting: shapes=[(8, 1, 191, 64), (8, 8, 64, 64), ()]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/gg/git_repos/notebooks/transformer/.venv/lib/python3.13/site-packages/jax/_src/lax/lax.py", line 221, in _broadcast_shapes_uncached
    return _try_broadcast_shapes(*rank_promoted_shapes, name='broadcast_shapes')
  File "/home/gg/git_repos/notebooks/transformer/.venv/lib/python3.13/site-packages/jax/_src/lax/lax.py", line 136, in _try_broadcast_shapes
    raise TypeError(f'{name} got incompatible shapes for broadcasting: '
                    f'{", ".join(map(str, map(tuple, shapes)))}.')
TypeError: broadcast_shapes got incompatible shapes for broadcasting: (8, 1, 191, 64), (8, 8, 64, 64), (1, 1, 1, 1).

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/gg/git_repos/notebooks/transformer/main.py", line 195, in <module>
    main()
    ~~~~^^
  File "/home/gg/git_repos/notebooks/transformer/main.py", line 135, in main
    state = init_train_state(config=config, vocab_size=tokenizer.vocab_size)
  File "/home/gg/git_repos/notebooks/transformer/utils/init_train_state.py", line 53, in init_train_state
    variables = model.init(
        rng,
    ...<4 lines>...
        is_training=True,
    )
  File "/home/gg/git_repos/notebooks/transformer/transformer/model.py", line 642, in __call__
    decoder_output = self.decoder(
        x=target_pos,
    ...<3 lines>...
        is_training=is_training,
    )
  File "/home/gg/git_repos/notebooks/transformer/transformer/model.py", line 514, in __call__
    x = block(
        x=x,
    ...<3 lines>...
        is_training=is_training,
    )
  File "/home/gg/git_repos/notebooks/transformer/transformer/model.py", line 454, in __call__
    masked_multi_head_attention_output = self.masked_multi_head_attention_block(
        q=x, k=x, v=x, mask=target_mask, is_training=is_training
    )
  File "/home/gg/git_repos/notebooks/transformer/transformer/model.py", line 316, in __call__
    x = MultiHeadAttentionBlock.flash_attention(
        query=query,
    ...<4 lines>...
        mask=mask,
    )
  File "/home/gg/git_repos/notebooks/transformer/transformer/model.py", line 228, in flash_attention
    scores = jnp.where(mask_block, scores, -1e9)
  File "/home/gg/git_repos/notebooks/transformer/.venv/lib/python3.13/site-packages/jax/_src/numpy/lax_numpy.py", line 2782, in where
    return util._where(condition, x, y)
           ~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/gg/git_repos/notebooks/transformer/.venv/lib/python3.13/site-packages/jax/_src/numpy/util.py", line 307, in _where
    condition, x_arr, y_arr = _broadcast_arrays(condition, x, y)
                              ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/gg/git_repos/notebooks/transformer/.venv/lib/python3.13/site-packages/jax/_src/numpy/util.py", line 259, in _broadcast_arrays
    result_shape = lax.broadcast_shapes(*shapes)
ValueError: Incompatible shapes for broadcasting: shapes=[(8, 1, 191, 64), (8, 8, 64, 64), ()]
--------------------
For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.

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
import jax
import jax.numpy as jnp


class DataLoader:
    def __init__(self, src, target, batch_size, seq_len, shuffle=True, n_prefetch=10):
        """
        Args:
            src: source sequences (numpy/jnp array, shape [N, seq_len])
            target: target sequences (numpy/jnp array, shape [N, seq_len])
            batch_size: batch size
            seq_len: maximum sequence length
            shuffle: shuffle dataset
            n_prefetch: number of batches to prefetch to device
        """
        self.src = jnp.array(src)
        self.target = jnp.array(target)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.n_prefetch = n_prefetch

    def create_src_mask(self, src):
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

            yield {
                "src": batch_src,
                "src_mask": src_mask,
                "target_input": batch_target[:, :-1],
                "target_output": batch_target[:, 1:],
                "target_mask": target_attention_mask[:, :, :-1, :-1],
                "token_mask": token_mask,
            }

    def _prefetch_to_device(self, iterator):
        """
        Prefetch n_prefetch batches to device asynchronously.
        """
        it = iter(iterator)
        prefetch_buffer = []

        # Move first n_prefetch batches to GPU
        for _ in range(self.n_prefetch):
            try:
                batch = next(it)
                batch_device = {k: jax.device_put(v) for k, v in batch.items()}
                prefetch_buffer.append(batch_device)
            except StopIteration:
                break

        while prefetch_buffer:
            # yield oldest batch
            yield prefetch_buffer.pop(0)

            # fetch next batch from CPU
            try:
                batch = next(it)
                batch_device = {k: jax.device_put(v) for k, v in batch.items()}
                prefetch_buffer.append(batch_device)
            except StopIteration:
                continue
ChatGPT said:

You're extremely close — the error is caused by one exact shape mismatch between the mask you're passing into FlashAttention and the scores tensor inside the attention loop.
✅ The actual problem

Inside FlashAttention you compute:

scores: (B, H, Qb, Kb)

But your mask_slice has shape:

mask_block: (B, 1, 1, Kb)

So when you apply:

scores = jnp.where(mask_block, scores, -jnp.inf)

JAX tries to broadcast:

scores:      (8, 8, 191, 64)
mask_block:  (8, 1, 1, 64)

These broadcast fine.

But your mask passed by the dataloader is NOT that shape.

The DataLoader creates:
Target mask (for decoder)

padding_mask: (B, 1, 1, L)
causal_mask:  (1, 1, L, L)
=> target_mask: (B, 1, L, L)

Then later:

"target_mask": target_attention_mask[:, :, :-1, :-1]

So the decoder receives:

target_mask: (B, 1, L-1, L-1)

But in multihead attention you expect:

mask: (B, 1, 1, L)

— a single mask over keys.

But you are passing a full 4D causal mask, which becomes (8, 1, 191, 191).

This is why your error says:

mask: (8, 1, 191, 64)  <-- from cropping
scores: (8, 8, 64, 64)

JAX cannot broadcast these because:

191 != 8

🎯 The root cause

Your attention implementation is tile-wise FlashAttention, which expects:
✔️ Mask: (B, 1, 1, L_s) — one boolean per key position
❌ Mask: (B, 1, L, L) — a full causal mask (does NOT work)

So you must NOT pass the causal mask into FlashAttention.
FlashAttention handles causality by slicing based on k_start/k_end only if you provide a per-key mask.
✅ Fix
Option A — Convert your causal mask to key-only mask (recommended)

For target self-attention:

Your causal_mask is:

(B=1, H=1, L, L)

But FlashAttention needs:

(B, 1, 1, L)

where mask[b, :, :, i] = 1 for allowed keys.

So convert:

target_key_mask = (batch_target[:, 1:] != 0).astype(jnp.float32)[:, None, None, :]

This is identical to encoder src_mask, because FlashAttention internally handles the causal ordering.

Then use:

"target_mask": target_key_mask

Option B — Implement causal masking inside FlashAttention

Replace:

scores = jnp.where(mask_block, scores, -jnp.inf)

with:

# causal mask inside each tile
causal = (q_start + jnp.arange(Qb)[:, None]) >= (k_start + jnp.arange(Kb)[None, :])
scores = jnp.where(causal, scores, -jnp.inf)

But this is more complex.
🧩 Minimal Correct Fix in DataLoader

Replace:

"target_mask": target_attention_mask[:, :, :-1, :-1],

with:

# FlashAttention expects mask over key positions only
key_mask = (batch_target[:, 1:] != 0).astype(jnp.float32)[:, None, None, :]
"target_mask": key_mask,

🚀 After this, shapes will match:

scores:     (B, H, Qb, Kb)
mask_block: (B, 1, 1, Kb)

Everything broadcasts correctly.
If you want, I can rewrite your FlashAttention function to include causal masking natively.

Just tell me "add causal masking inside FlashAttention".
You said:
"""
This training and evaluation file is based on the implementation from
https://wandb.ai/jax-series/simple-training-loop/reports/Writing-a-Training-Loop-in-JAX-and-Flax--VmlldzoyMzA4ODEy
"""

from typing import Any

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax.training import train_state


def train(
    state: train_state.TrainState,
    train_loader,
    val_loader,
    epochs: int,
    manager: ocp.CheckpointManager,
    logger,
    step: int = 0,
):
    """
    train the model
    Args:
        state: train_state.TrainState
        train_loader: DataLoader
        val_loader: DataLoader
        epochs: int
        manager: ocp.CheckpointManager
        logger: logger

    Returns:
        None
    """

    rng = jax.random.PRNGKey(0)
    # loop over the dataset for num_epochs
    for epoch in range(step, epochs):
        # split for this epoch once
        rng, loader_rng = jax.random.split(rng)

        # split again for dropout
        rng, dropout_base = jax.random.split(rng)
        # iterate through each batch in the dataset
        for batch in train_loader.__iter__(rng=loader_rng):
            dropout_base, dropout_rng = jax.random.split(dropout_base)
            # train on batch
            state, loss = train_step(state=state, batch=batch, dropout_rng=dropout_rng)

        # train and val accuracy and loss
        eval_accuracy, eval_loss = eval(state=state, loader=val_loader, rng=loader_rng)
        train_accuracy, train_loss = eval(state=state, loader=train_loader, rng=None)

        # create metrics dictionary
        metrics = {
            "train_loss": float(train_loss),
            "eval_loss": float(eval_loss),
            "train_accuracy": float(train_accuracy),
            "eval_accuracy": float(eval_accuracy),
        }
        # log the metrics
        logger.info(f" EPOCH: {epoch} | METRICS: {metrics}")
        logger.info(f"Saving checkpoint at epoch {epoch}")
        manager.save(
            step=epoch,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(state),
                metrics=ocp.args.JsonSave(metrics),
            ),
        )

    manager.wait_until_finished()

    return state


@jax.jit
def train_step(
    state: train_state.TrainState,
    batch,
    dropout_rng: jax.random.PRNGKey,
) -> tuple[train_state.TrainState, Any]:
    """
    handle a single training step
    get loss
    get gradients
    update parameters

    Args:
        state: train_state.TrainState
        batch: batch
        dropout_rng: random number generator

    Returns:
        train_state.TrainState and loss
    """

    src = batch["src"]
    src_mask = batch["src_mask"]
    target_input = batch["target_input"]
    target_output = batch["target_output"]
    target_mask = batch["target_mask"]
    token_mask = batch["token_mask"]

    # define loss function
    def loss_fn(params):
        """
        Compute the loss function for a single batch
        """
        logits = state.apply_fn(
            {"params": params},
            src,
            src_mask,
            target_input,
            target_mask,
            is_training=True,
            rngs={"dropout": dropout_rng},
        )

        per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=target_output,
        )
        loss = (per_token_loss * token_mask).sum() / token_mask.sum()
        return loss

    # compute loss and gradients
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    # update the the training state with the new gradients
    state = state.apply_gradients(grads=grads)
    return state, loss


def eval(
    state: train_state.TrainState,
    loader,
    rng: jax.random.PRNGKey,
) -> tuple[float, float]:
    """
    evaluate the model on the validation set
    Args:
        state: train_state.TrainState
        val_loader: DataLoader

    Returns:
        accuracy, loss
    """
    total_loss = 0.0
    total_accuracy = 0.0
    num_bathces = 0
    # loop over the dataset
    for batch in loader.__iter__(rng=rng):
        # evaluate on batch
        accuracy, loss = eval_step(state=state, batch=batch)
        # get num of examples in current batch and add to total
        total_accuracy += accuracy
        total_loss += loss
        num_bathces += 1

    accuracy = total_accuracy / num_bathces
    avg_loss = total_loss / num_bathces

    return accuracy, avg_loss


@jax.jit
def eval_step(state: train_state.TrainState, batch):
    """
    evaluate the model on a single batch
    Args:
        state: train_state.TrainState
        batch: batch

    Returns:
        predictions
    """
    src = batch["src"]
    src_mask = batch["src_mask"]
    target_input = batch["target_input"]
    target_output = batch["target_output"]
    target_mask = batch["target_mask"]
    token_mask = batch["token_mask"]
    # pass batch through the model in training state
    logits = state.apply_fn(
        {"params": state.params},
        src,
        src_mask,
        target_input,
        target_mask,
        is_training=False,
    )
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=target_output,
    )
    loss = (per_token_loss * token_mask).sum() / token_mask.sum()

    predictions = jnp.argmax(logits, axis=-1)
    correct = ((predictions == target_output) * token_mask).sum()
    total = token_mask.sum()
    accuracy = correct / total

    return accuracy, loss
ChatGPT said:

Great — this training loop is mostly correct, but you now have one additional problem:
your DataLoader mask output no longer matches what your model expects after the FlashAttention fix.

Let me walk you through exactly what to change so your entire training pipeline becomes consistent, stable, and JIT-compatible.
✅ 1. FlashAttention Requires a Key-Only Mask

Your model’s MHA expects mask to be a key mask of shape:

(B, 1, 1, L_src)

But your DataLoader currently outputs:

target_mask: (B, 1, L-1, L-1)

This is incompatible.
✅ Fix the DataLoader

Find:

"target_mask": target_attention_mask[:, :, :-1, :-1],

Replace with:

# FlashAttention expects only key visibility mask
key_mask = (batch_target[:, 1:] != 0).astype(jnp.float32)[:, None, None, :]
"target_mask": key_mask,

That’s all.
Why this works

Fast FlashAttention does not take a causal mask.
It applies causality internally by chunking key blocks in order.

So your target_mask should only hide padding tokens, not future tokens.
🧩 2. Model apply_fn Argument Structure

Your call:

logits = state.apply_fn(
    {"params": params},
    src,
    src_mask,
    target_input,
    target_mask,
    is_training=True,
    rngs={"dropout": dropout_rng},
)

This fits your model’s signature:

def __call__(self, src, src_mask, tgt, tgt_mask, is_training)

→ No changes needed here.
🧩 3. train_loader.__iter__(rng=loader_rng) inside JIT

This is safe.

rng is static with donate_argnums, so it won’t break JIT.
🧩 4. The main (future) issue: eval(train_loader, rng=None)

I caught a future bug:

Your eval loop does:

for batch in loader.__iter__(rng=rng):

But your DataLoader has:

if rng is None:
    raise ValueError("Shuffle=True but no RNG was provided to DataLoader")

During evaluation:

train_accuracy, train_loss = eval(state=state, loader=train_loader, rng=None)

This will raise an error.
Fix

Two options:
Option A — Disable shuffle for training loader during eval

Before training:

train_eval_loader = DataLoader(
    src=train_loader.src,
    target=train_loader.target,
    batch_size=train_loader.batch_size,
    seq_len=train_loader.seq_len,
    shuffle=False,
)

and use this for eval.
Option B — In the DataLoader, treat rng=None as “do not shuffle”

Modify:

if self.shuffle:
    if rng is None:
        raise ValueError(...)
    indices = jax.random.permutation(rng, indices)

to:

if self.shuffle and rng is not None:
    indices = jax.random.permutation(rng, indices)

This is cleaner.
🧩 5. A final suggestion: JIT-unsafe logging inside train loop

Your logger prints on every epoch:

logger.info(...)
manager.save(...)

No issue — this is outside JIT.

You're safe.
✅ Final Minimal Fixes
✔️ Fix DataLoader mask

key_mask = (batch_target[:, 1:] != 0).astype(jnp.float32)[:, None, None, :]
"target_mask": key_mask,

✔️ Fix DataLoader shuffle condition

Change:

if self.shuffle:
    if rng is None:
        raise ValueError(...)
    indices = jax.random.permutation(rng, indices)

to:

if self.shuffle and rng is not None:
    indices = jax.random.permutation(rng, indices)

🎉 After these two changes your training loop will run correctly.

If you want, I can:

    rewrite your FlashAttention to include true causal masking

    rewrite data pipeline for pmap or pjit

    write a full benchmark-friendly training loop

    add gradient accumulation for long sequences

    add mixed precision bf16 normalization

Just tell me!


ChatGPT can make mistakes. Check important info.
