import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from lightning import LightningDataModule


class PromotersDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 1024,
        num_workers: int = 1,
        pin_memory: bool = False,
        test_size: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters(dict(dataset=self.__class__.__name__))
        self.save_hyperparameters()

        # Download and setup data
        # TODO: move the dataset processing part here / call an helper function
        # This step (for now) ends up with a one-hot encoded tensor dataset
        print("Skipping data setup for now")

        # Load dataset Tensors
        path = os.path.join(os.getcwd(), "data", "promoters", "dataset.npz")
        print(f"Extracting compressed arrays from {path}...")
        compressed_data = np.load(path)
        data = torch.as_tensor(compressed_data["data"], dtype=torch.float32)
        labels = torch.as_tensor(compressed_data["labels"], dtype=torch.float32)
        print("Data:", data.shape, data.dtype)
        print("Labels:", labels.shape, labels.dtype)

        # train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data, labels, test_size=test_size, stratify=labels, random_state=seed
        )

    def train_dataloader(self):
        return DataLoader(
            TensorDataset(self.X_train, self.y_train),
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            TensorDataset(self.X_test, self.y_test),
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        # TODO: implement val-test split (?)
        return self.val_dataloader()


class DHSDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 1024,
        num_workers: int = 1,
        pin_memory: bool = False,
        test_size: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters(dict(dataset=self.__class__.__name__))
        self.save_hyperparameters()

        # Download and setup data
        # TODO: move the dataset processing part here / call an helper function
        # This step (for now) ends up with a one-hot encoded tensor dataset
        print("Skipping data setup for now")

        # Load dataset Tensors
        path = os.path.join(os.getcwd(), "data", "dhs", "dataset.npz")
        print(f"Extracting compressed arrays from {path}...")
        compressed_data = np.load(path)
        data = torch.as_tensor(compressed_data["data"], dtype=torch.float32)
        labels = torch.as_tensor(compressed_data["labels"], dtype=torch.float32)
        print("Data:", data.shape, data.dtype)
        print("Labels:", labels.shape, labels.dtype)

        # train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data, labels, test_size=test_size, stratify=labels, random_state=seed
        )

    def train_dataloader(self):
        return DataLoader(
            TensorDataset(self.X_train, self.y_train),
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            TensorDataset(self.X_test, self.y_test),
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        # TODO: implement val-test split (?)
        return self.val_dataloader()
