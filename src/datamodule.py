import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from lightning import LightningDataModule
from tqdm import tqdm


class TestDataModule(LightningDataModule):
    def __init__(
        self,
        path: str,
        batch_size: int = 256,
        workers: int = 1,
        small: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str):
        # Download and setup data
        # TODO: move the dataset processing part here / call an helper function
        # This step (for now) ends up with a one-hot encoded tensor dataset
        print("Skipping data setup for now")

        # Train-Test split
        # TODO: split the dataset into train and test sets (stratified)
        # if we integrate with wandb this step can be done in the artifact setup
        print("Skipping train-test for now")

        # Load dataset Tensors
        print("Extracting compressed arrays...")
        compressed_data = np.load(self.hparams["path"])
        self.data = torch.as_tensor(compressed_data["data"], dtype=torch.float32)
        self.labels = torch.as_tensor(compressed_data["labels"], dtype=torch.float32)
        if self.hparams["small"]:
            idxs = torch.randint(0, len(self.data), (10000,))
            self.data = self.data[idxs]
            self.labels = self.labels[idxs]
        print("Done!")

    def train_dataloader(self):
        return DataLoader(
            TensorDataset(self.data, self.labels),
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["workers"],
            shuffle=True,
        )

    def val_dataloader(self):
        # TODO: implement test split
        idxs = torch.randint(0, len(self.data), (1000,))
        return DataLoader(
            TensorDataset(self.data[idxs], self.labels[idxs]),
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["workers"],
        )

    def test_dataloader(self):
        # TODO: implement test split
        return self.train_dataloader()
