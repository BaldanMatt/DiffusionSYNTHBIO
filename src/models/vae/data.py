import polars as pl
import numpy as np
import torch
from tqdm import tqdm


class DNADataset(torch.utils.data.Dataset):
    def __init__(self, path: str):
        print(f"Loading data from {path}...")
        df = pl.read_csv(path)
        print("Processing data...")

        # load and process sequences
        self.data = df["Sequence"].to_numpy().astype(str)
        sequence_length = len(self.data[0])
        chars = self.data.view("S1").reshape(-1, sequence_length, 4)[..., 0]
        self.char_list = np.unique(chars)
        self.data_one_hot = self.one_hot_encode(chars, self.char_list)

        # load and process labels
        self.labels = df["species"].to_numpy().astype(str)
        self.label_list = np.unique(self.labels)
        self.labels_one_hot = self.one_hot_encode(self.labels, self.label_list)

        print("Done!")

    def __len__(self):
        return len(self.data_one_hot)

    def __getitem__(self, idx):
        return self.data_one_hot[idx], self.labels_one_hot[idx]

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


class DNADiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, autoencoder, device=None, batch_size=1):
        encoded_mu, encoded_sigma = [], []
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        autoencoder = autoencoder.to(device)
        for X, y in tqdm(dataloader, desc="Pre encoding data"):
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
