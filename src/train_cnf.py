import os
from lightning import Trainer
from lightning.pytorch import callbacks, loggers

from datamodule import TestDataModule
from diffusion import DiffusionTransformer


config: dict = dict(
    input_dim=5,
    cond_dim=16,
    hidden_dim=128,
    depth=16,
    num_heads=8,
    patch_size=4,
    x_jitter_std=0.01,
    learning_rate=1e-5,
    weight_decay=1e-3,
)

if __name__ == "__main__":
    model = DiffusionTransformer(**config)
    datamodule = TestDataModule(
        os.getcwd() + "/data/dataset_compressed.npz", batch_size=1024
    )

    logger = loggers.WandbLogger(project="DNAdiffusion", log_model=True)
    trainer = Trainer(
        max_epochs=100,
        logger=logger,
        gradient_clip_val=0.5,
        callbacks=[
            callbacks.ModelCheckpoint(save_last=True),
            callbacks.RichModelSummary(),
            callbacks.RichProgressBar(),
        ],
    )
    trainer.fit(model, datamodule=datamodule)
