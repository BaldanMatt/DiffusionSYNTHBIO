import os
import numpy as np
import polars as pl
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from lightning import LightningDataModule


def one_hot_encode(array):
    unique_values = np.unique(array)
    masks = [array == val for val in unique_values]
    nums = np.select(masks, list(range(len(masks))))
    one_hot = np.eye(len(masks), dtype=bool)[nums]
    return one_hot, unique_values


class EPDGenDNA_256(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 256,
        num_workers: int = 1,
        pin_memory: bool = False,
        test_size: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters(dict(dataset=self.__class__.__name__))
        self.save_hyperparameters()

        # Download data
        df = pl.read_csv('hf://datasets/Zehui127127/latent-dna-diffusion/sequence_central_256.csv')
            
        # Preprocess
        sequences = df['Sequence'].to_numpy()
        sequences = np.array([list(s) for s in sequences], dtype='|S1')
        sequences_onehot, self.x_descr = one_hot_encode(sequences)

        species = df['species'].to_numpy()
        species_onehot, self.y_descr = one_hot_encode(species)

        # train-test split
        x = torch.as_tensor(sequences_onehot, dtype=torch.float32)
        y = torch.as_tensor(species_onehot, dtype=torch.float32)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=test_size, stratify=y, random_state=seed
        )

    def train_dataloader(self):
        return DataLoader(
            TensorDataset(self.x_train, self.y_train),
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            TensorDataset(self.x_test, self.y_test),
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        # TODO: implement val-test split (?)
        return self.val_dataloader()
    

class EPDGenDNA_2048(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 256,
        num_workers: int = 1,
        pin_memory: bool = False,
        test_size: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters(dict(dataset=self.__class__.__name__))
        self.save_hyperparameters()

        # Download data
        df = pl.read_csv('hf://datasets/Zehui127127/latent-dna-diffusion/sequence.csv')
            
        # Preprocess
        sequences = df['Sequence'].to_numpy()
        sequences = np.array([list(s) for s in sequences], dtype='|S1')
        sequences_onehot, self.x_descr = one_hot_encode(sequences)

        species = df['species'].to_numpy()
        species_onehot, self.y_descr = one_hot_encode(species)

        # train-test split
        x = torch.as_tensor(sequences_onehot, dtype=torch.float32)
        y = torch.as_tensor(species_onehot, dtype=torch.float32)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=test_size, stratify=y, random_state=seed
        )

    def train_dataloader(self):
        return DataLoader(
            TensorDataset(self.x_train, self.y_train),
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            TensorDataset(self.x_test, self.y_test),
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        # TODO: implement val-test split (?)
        return self.val_dataloader()
    

class DHS(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 256,
        num_workers: int = 1,
        pin_memory: bool = False,
        test_size: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters(dict(dataset=self.__class__.__name__))
        self.save_hyperparameters()

        # Download data
        compressed = np.load(os.path.join(os.getcwd(), self.__class__.__name__, "DHS_one_hot_centered.npz"))
        sequences_onehot = compressed["X"]
        classes_onehot = compressed["y"]

        # TODO: check this is correct
        self.x_descr = [b"A", b"C", b"G", b"T", b"N"]
        self.y_descr = [f"class {i}" for i in range(classes_onehot.shape[-1])]
            
        # train-test split
        x = torch.as_tensor(sequences_onehot, dtype=torch.float32)
        y = torch.as_tensor(classes_onehot, dtype=torch.float32)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=test_size, stratify=y, random_state=seed
        )

    def train_dataloader(self):
        return DataLoader(
            TensorDataset(self.x_train, self.y_train),
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            TensorDataset(self.x_test, self.y_test),
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        # TODO: implement val-test split (?)
        return self.val_dataloader()