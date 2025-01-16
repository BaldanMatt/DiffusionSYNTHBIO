import polars as pl
import numpy as np
from torch.utils.data import Dataset


class DNADataset(Dataset):
    def __init__(self, path: str):
        df = pl.read_csv(path)
        self.data = df["Sequence"].to_numpy().astype(str)
        self.labels = df["species"].to_numpy().astype(str)

        self.one_hot = self.one_hot_encode(self.data)

    def __len__(self):
        return len(self.one_hot)

    def __getitem__(self, idx):
        return self.one_hot[idx]

    @staticmethod
    def one_hot_encode(data):
        sequence_length = len(data[0])
        chars = data.view("S1").reshape(-1, sequence_length, 4)[..., 0]
        masks = [chars == b"A", chars == b"C", chars == b"G", chars == b"T"]
        nums = np.select(masks, [0, 1, 2, 3], default=4)
        one_hot = np.eye(5)[nums]
        return one_hot

    @staticmethod
    def one_hot_decode(one_hot):
        nums = np.select(one_hot.T.astype(bool), [0, 1, 2, 3, 4]).T
        chars = np.array([b"A", b"C", b"G", b"T", b"N"])[nums]
        return chars
