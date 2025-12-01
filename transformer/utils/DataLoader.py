import jax
import jax.numpy as jnp
from flax.training.common_utils import shard
import itertools


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
        return (src != 0).astype(jnp.bfloat16)[
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
        padding_mask = (target != 0).astype(jnp.bfloat16)[:, None, None, :]  # (B,1,1,L)
        # create causal mask (1,L,L) (lower triangular matrix)
        causal_mask = jnp.tril(
            jnp.ones((self.seq_len, self.seq_len), dtype=jnp.bfloat16)
        )[None, None, :, :]
        # create final mask where future tokens are masked and padding tokens are ignored
        return padding_mask * causal_mask  # (B,1,L,L)

    def __iter__(self, rng=None):
        return self._prefetch_to_device(self._batch_generator(rng))

    def _batch_generator(self, rng):
        N = self.src.shape[0]
        indices = jnp.arange(N)

        # Use provided RNG for shuffling
        if self.shuffle:
            if rng is None:
                rng = jax.random.PRNGKey(0)
            indices = jax.random.permutation(rng, indices)
        for start in range(0, N, self.batch_size):
            end = start + self.batch_size
            batch_idx = indices[start:end]
            if len(batch_idx) < self.batch_size:
                break

            batch_src = jnp.array(self.src[batch_idx])
            batch_target = jnp.array(self.target[batch_idx])

            src_mask = self.create_src_mask(batch_src)
            target_mask = self.create_target_mask(batch_target)

            batch = {
                "src": batch_src,  # (b, seq_len)
                "src_mask": src_mask,  # (b, 1, 1, seq_len)
                "target_input": batch_target[
                    :, :-1
                ],  # exclude last token (b, seq_len-1)
                "target_output": batch_target[
                    :, 1:
                ],  # exclude first token (b, seq_len-1)
                "target_mask": target_mask[
                    :, :, :-1, :-1
                ],  # (b, 1, seq_len-1, seq_len-1)
            }
            yield batch

    def _prefetch_to_device(self, iterator):
        """
        Prefetch `n_prefetch` batches to device asynchronously.
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
