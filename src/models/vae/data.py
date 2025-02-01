import polars as pl
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from lightning import LightningDataModule


class TestDataModule(LightningDataModule):
    def __init__(self, path: str, batch_size: int = 256):
        super().__init__()
        self.path = path
        self.batch_size = batch_size

    def setup(self, stage: str):
        print(f"Loading data from {self.path}...")
        df = pl.read_csv(self.path)
        print("Processing data...")

        # process sequences
        self.data = df["Sequence"].to_numpy().astype(str)
        sequence_length = len(self.data[0])
        chars = self.data.view("S1").reshape(-1, sequence_length, 4)[..., 0]
        self.char_list = np.unique(chars)
        self.data_one_hot = self.one_hot_encode(chars, self.char_list)

        # process labels
        self.labels = df["species"].to_numpy().astype(str)
        self.label_list = np.unique(self.labels)
        self.labels_one_hot = self.one_hot_encode(self.labels, self.label_list)
        print("Done!")

    def train_dataloader(self):
        dataset = TensorDataset(torch.as_tensor(self.data_one_hot))
        return DataLoader(dataset, self.batch_size, shuffle=True, num_workers=1)

    @staticmethod
    def one_hot_encode(data, classes):
        masks = [data == v for v in classes]
        nums = np.select(masks, list(range(len(classes))))
        one_hot = np.eye(len(classes), dtype=np.float32)[nums]
        return one_hot

    @staticmethod
    def one_hot_decode(one_hot, classes):
        nums = np.select(one_hot.T.astype(bool), list(range(len(classes)))).T
        return classes[nums]

    def find_subsequence(self, one_hot, subsequence: str):
        masks = {k: one_hot[..., i].astype(bool) for i, k in enumerate(self.char_list)}
        masks[b"W"] = masks[b"A"] | masks[b"T"]
        matches = np.ones_like(one_hot[..., 0], dtype=bool)
        for i, char in enumerate(subsequence):
            matches &= np.roll(masks[char.encode("utf-8")], -i, axis=-1)
        return matches
