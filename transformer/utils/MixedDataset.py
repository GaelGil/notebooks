import grain
from utils.Source import Source
import random


class MixedDataset(grain.DataLoader):
    """
    Combines English and Nahuatl data sources for Phase 2 training.

    Uses 80/20 ratio: 80% Spanish->Nahuatl, 20% Spanish->English
    to prevent catastrophic forgetting of English translation.
    """

    def __init__(
        self,
        en_data: Source,
        nah_data: Source,
        seed: int = 42,
    ):
        self.en_data = en_data
        self.nah_data = nah_data

        self.n_nah = len(self.nah_data)
        self.n_en = len(self.en_data)

        rng = random.Random(seed)

        self.en_indices = rng.sample(range(self.n_en), self.n_nah)

    def __len__(self):
        return self.n_nah * 2

    def __getitem__(self, idx):
        if idx < self.n_nah:
            return self.nah_data[idx]
        else:
            en_idx = self.en_indices[idx - self.n_nah]
            return self.en_data[en_idx]
