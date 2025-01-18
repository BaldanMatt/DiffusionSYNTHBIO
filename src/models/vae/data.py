import polars as pl
import numpy as np
import torch
from tqdm import tqdm


class DNADataset(torch.utils.data.Dataset):
    def __init__(self, path: str):
        print(f"Loading data from {path}...")
        df = pl.read_csv(path)
        print("Processing data...")
        self.data = df["Sequence"].to_numpy().astype(str)
        self.labels = df["species"].to_numpy().astype(str)
        self.data_one_hot = self.one_hot_encode(self.data)
        print("Done!")

    def __len__(self):
        return len(self.data_one_hot)

    def __getitem__(self, idx):
        return self.data_one_hot[idx]

    @staticmethod
    def one_hot_encode(data):
        sequence_length = len(data[0])
        chars = data.view("S1").reshape(-1, sequence_length, 4)[..., 0]
        masks = [chars == b"T", chars == b"A", chars == b"C", chars == b"G"]
        nums = np.select(masks, [0, 1, 2, 3], default=4)
        one_hot = np.eye(5, dtype=np.float32)[nums]
        return one_hot

    @staticmethod
    def one_hot_decode(one_hot):
        nums = np.select(one_hot.T.astype(bool), [0, 1, 2, 3, 4]).T
        chars = np.array([b"T", b"A", b"C", b"G", b"N"])[nums]
        return chars

    @staticmethod
    def find_subsequence(one_hot, subsequence: str):
        masks = {
            "T": one_hot[..., 0].astype(bool),
            "A": one_hot[..., 1].astype(bool),
            "C": one_hot[..., 2].astype(bool),
            "G": one_hot[..., 3].astype(bool),
            "N": one_hot[..., 4].astype(bool),
        }
        masks["W"] = masks["A"] | masks["T"]
        matches = np.ones_like(one_hot[..., 0], dtype=bool)
        for i, char in enumerate(subsequence):
            matches &= np.roll(masks[char], -i, axis=-1)
        return matches


class DNADiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, autoencoder, device=None, batch_size=1):
        encoded_mu, encoded_sigma = [], []
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        autoencoder = autoencoder.to(device)
        for X in tqdm(dataloader, desc="Pre encoding data"):
            mu, sigma = autoencoder.encode(X.to(device=device))
            encoded_mu.append(mu.detach().cpu().numpy())
            encoded_sigma.append(sigma.detach().cpu().numpy())
        self.mu = np.concatenate(encoded_mu)
        self.sigma = np.concatenate(encoded_sigma)

    def __len__(self):
        return len(self.mu)

    def __getitem__(self, idx):
        mu, sigma = self.mu[idx], self.sigma[idx]
        x1 = mu + sigma * np.random.randn(*mu.shape)
        x0 = np.random.randn(*mu.shape)
        t = np.random.rand()
        return t, x0, x1
