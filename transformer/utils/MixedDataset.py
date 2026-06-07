import grain
from utils.Source import Source


class MixedDataset(grain.DataLoader):
    """
    Combines English and Nahuatl data sources for Phase 2 training.

    The ratio means:
    - keep all Nahuatl samples
    - add enough English samples to make the final dataset match the ratio

    Example:
    nah_ratio = 0.8
    1000 Nahuatl samples -> add 250 English samples
    final dataset = 1000 Nahuatl + 250 English = 80/20
    """

    def __init__(
        self,
        en_data: Source,
        nah_data: Source,
        nah_ratio: float = 0.8,
    ):
        if not 0 < nah_ratio < 1:
            raise ValueError("nah_ratio must be between 0 and 1")

        self.en_data = en_data
        self.nah_data = nah_data

        self.n_nah = len(nah_data)
        self.n_en = len(en_data)

        self.n_en_samples = int(self.n_nah * (1 - nah_ratio) / nah_ratio)

    def __len__(self):
        return self.n_nah + self.n_en_samples

    def __getitem__(self, idx):
        if idx < self.n_nah:
            return self.nah_data[idx]

        en_idx = (idx - self.n_nah) % self.n_en
        return self.en_data[en_idx]
