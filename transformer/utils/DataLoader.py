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

    def padding_mask(self, seq: jnp.ndarray):
        """
        Create padding mask to ignore padded tokens.
        seq is a sequence of shape (B, seq_len)

        ie:
        seq = [1, 2, 3, 0, 0, 0]
        mask = [1, 1, 1, 0, 0, 0]
        Args:
            seq: sequence
        Returns:
            padding mask with shape (N, 1, 1, seq_len)
        """
        return (seq != 0).astype(jnp.float32)[:, None, None, :]

    def causal_mask(self, padding_mask) -> jnp.ndarray:
        """
        Create causal mask to ignore future tokens in target sequence.

        ie: seq_len = 3, max_seq_len = 6
        padding_mask = [1, 1, 1, 0, 0, 0]
        causal_mask = [[1, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0],
                       [1, 1, 1, 0, 0, 0],
                       [1, 1, 1, 1, 0, 0],
                       [1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1]]
        final_mask = padding_mask * causal_mask = [[1, 0, 0, 0, 0, 0],
                                                  [1, 1, 0, 0, 0, 0],
                                                  [1, 1, 1, 0, 0, 0],
                                                  [1, 1, 1, 0, 0, 0],
                                                  [1, 1, 1, 0, 0, 0],
                                                  [1, 1, 1, 0, 0, 0]]
        Args:
            padding_mask: padding mask
        Returns:
            jnp.ndarray
        """
        # create a matrix of shape (L_target, L_target) then apply tril to get the lower triangular matrix
        causal_mask = jnp.tril(
            jnp.ones((self.seq_len, self.seq_len), dtype=jnp.float32)
        )[None, None, :, :]

        # multiply the each row of the mask by the padding mask to get the final mask
        # this will ignore the future tokens in the target sequence
        return padding_mask * causal_mask

    def __iter__(self, rng=None):
        return self._prefetch_to_device(self._batch_generator(rng))

    def _batch_generator(self, rng):
        N = self.src.shape[0]
        indices = jnp.arange(N)

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
            src_mask = self.padding_mask(batch_src)

            batch_target = jnp.array(self.target[batch_idx])
            target_input = batch_target[:, :-1]  # all tokens except the last
            target_output = batch_target[:, 1:]  # all tokens except the first
            target_mask = self.padding_mask(target_input)
            target_output_mask = self.padding_mask(target_output)
            # target_mask = self.causal_mask(
            #     self.padding_mask(batch_target), L_target=batch_target.shape[1]
            # )

            yield {
                "src_input": batch_src,
                "src_mask": src_mask,
                "target_input": target_input,
                "target_mask": target_mask,
                "target_output": target_output,
                "target_output_mask": target_output_mask,
            }

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


# loader = DataLoader(
#     src=jnp.array([1, 2, 3, 4, 5, 6]),
#     target=jnp.array([1, 2, 3, 4, 5, 6]),
#     batch_size=2,
#     shuffle=True,
#     seq_len=6,
#     n_prefetch=1,
# )

# batch = jnp.array([[1, 2, 3, 0, 0, 0], [1, 2, 3, 0, 0, 0]])

# padding_mask = loader.padding_mask(batch)
# print(f"Batch: \n {batch}")
# print(f"Padding mask: \n {padding_mask}")
# print()
# causal_mask = loader.causal_mask(padding_mask)

# print("Causal mask: \n")
# print(causal_mask)

# print(f"padding mask shape: {padding_mask.shape}")
# print(f"causal mask shape: {causal_mask.shape}")
