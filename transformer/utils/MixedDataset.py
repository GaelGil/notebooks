import grain
import numpy as np
from jax import numpy as jnp


class MixedDataset(grain.DataLoader):
    """Combines English and Nahuatl data sources for Phase 2 training.

    Uses 80/20 ratio: 80% Spanish->Nahuatl, 20% Spanish->English
    to prevent catastrophic forgetting of English translation.
    """

    def __init__(
        self,
        en_data,  # Spanish-English data (20%)
        nah_data,  # Spanish-Nahuatl data (80%)
        mix_ratio: float = 0.8,
        seed: int = 42,
    ):
        self.en_data = en_data
        self.nah_data = nah_data
        self.mix_ratio = mix_ratio
        self.rng = np.random.default_rng(seed=seed)

        # Calculate sizes
        total_en = len(en_data)
        total_nah = len(nah_data)

        # Generate mixed indices - 80% Nahuatl, 20% English
        # Create enough indices for full epoch coverage
        max_size = max(total_en, total_nah) * 2
        self.indices = []

        for _ in range(max_size):
            if self.rng.random() < self.mix_ratio:
                self.indices.append(self.rng.integers(0, total_nah))
            else:
                self.indices.append(total_nah + self.rng.integers(0, total_en))

        self.indices = np.array(self.indices)
        self.nah_offset = total_nah

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        # If index < nah_offset, it's Nahuatl data
        # If index >= nah_offset, it's English data
        if real_idx < self.nah_offset:
            return self.nah_data[real_idx]
        else:
            return self.en_data[real_idx - self.nah_offset]
