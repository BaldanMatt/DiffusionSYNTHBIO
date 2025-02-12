import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from lightning import LightningDataModule
from tqdm import tqdm


class TestDataModule(LightningDataModule):
    def __init__(self, path: str, batch_size: int = 256, workers: int = 1):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.workers = workers

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
        compressed_data = np.load(self.path)
        self.data = torch.as_tensor(compressed_data["data"])
        self.labels = torch.as_tensor(compressed_data["labels"])
        print("Done!")

    def train_dataloader(self):
        dataset = TensorDataset(self.data, self.labels)
        return DataLoader(
            dataset, self.batch_size, shuffle=True, num_workers=self.workers
        )

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError
