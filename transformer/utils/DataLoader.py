import grain
from jax import numpy as jnp


class Source(grain.Dataloader):
    def __init__(self, src_path, tgt_path):
        self.src = jnp.load(src_path)
        self.tgt = jnp.load(tgt_path)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]
