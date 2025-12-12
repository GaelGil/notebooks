import grain
from jax import numpy as jnp


class Source(grain.DataLoader):
    def __init__(self, src_path, target_path):
        self.src = jnp.load(src_path)
        self.target = jnp.load(target_path)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.target[idx]
