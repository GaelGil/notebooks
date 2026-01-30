import grain
import numpy as np


class Source(grain.sources.RandomAccessDataSource):
    """
    Source data loader
    """

    def __init__(self, src_path: str, target_path: str, pad_id: int):
        """
        Args:
            src_path: path to the source data
            target_path: path to the target data
            pad_id: pad id

        Returns:
            None
        """
        self.src = np.load(src_path)
        self.target = np.load(target_path)
        self.pad_id = pad_id

    def __len__(self):
        """
        Returns:
            length of the source data
        """
        return len(self.src)

    def __getitem__(self, idx):
        return
