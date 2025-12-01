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
        self.src = src
        self.target = target
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.n_prefetch = n_prefetch

    def create_src_mask(self, src):
        return (src != 0).astype(jnp.float32)  # 1 for tokens, 0 for padding

    def prepare_encoder_mask(self, mask):
        return mask[:, None, None, :]  # (B, 1, 1, seq_len)

    def create_tgt_mask(self, target):
        batch_size, seq_len = target.shape
        padding_mask = (target != 0).astype(jnp.float32)[:, None, None, :]  # (B,1,1,L)
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bfloat16))[
            None, None, :, :
        ]
        return padding_mask * causal_mask  # (B,1,L,L)

    def __iter__(self):
        return self._prefetch_to_device(self._batch_generator())

    def _batch_generator(self):
        N = self.src.shape[0]
        indices = jnp.arange(N)
        if self.shuffle:
            indices = jax.random.permutation(jax.random.PRNGKey(0), indices)

        for start in range(0, N, self.batch_size):
            end = start + self.batch_size
            batch_idx = indices[start:end]
            if len(batch_idx) < self.batch_size:
                break

            batch_src = self.src[batch_idx]
            batch_target = self.target[batch_idx]

            src_mask = self.prepare_encoder_mask(self.create_src_mask(batch_src))
            target_attention_mask = self.create_tgt_mask(batch_target)
            token_mask = (batch_target[:, 1:] != 0).astype(
                jnp.bfloat16
            )  # (b, seq_len-1)
            batch = {
                "src": batch_src,
                "src_mask": src_mask,
                "target_input": batch_target[:, :-1],
                "target_output": batch_target[:, 1:],
                "target_mask": target_attention_mask[:, :, :-1, :-1],
                "token_mask": token_mask,
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
