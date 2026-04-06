import grain
from utils.Source import Source


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
        batch_size: int,
    ):
        self.en_data = en_data
        self.nah_data = nah_data
        nah_batches = int(len(nah_data) / batch_size)

        self.indices = list(range(nah_batches) * 2)

    def __len__(self):
        return self.indices

    def __getitem__(self, idx):
        if idx < int(self.indices / 2):
            return self.nah_data[idx]
        else:
            return self.en_data[idx]
