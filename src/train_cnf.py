import os
from lightning import Trainer
from lightning.pytorch import callbacks, loggers

from datamodule import TestDataModule
from diffusion import DiffusionTransformer


config: dict = dict(
    input_dim=5,
    cond_dim=16,
    hidden_dim=8 * 32,
    num_heads=8,
    depth=8,
    patch_size=4,
    cond_drop_prob=0.1,
    x_jitter_std=0.01,
    learning_rate=1e-4,
    weight_decay=1e-5,
)

if __name__ == "__main__":
    model = DiffusionTransformer(**config)
    datamodule = TestDataModule(
        os.getcwd() + "/data/dataset_compressed.npz", batch_size=1024
    )

    logger = loggers.WandbLogger(project="DNAdiffusion", log_model=True)
    trainer = Trainer(
        max_epochs=10,
        logger=logger,
        gradient_clip_val=1.0,
        callbacks=[
            callbacks.ModelCheckpoint(save_last=True, save_weights_only=True),
            callbacks.RichModelSummary(),
            callbacks.RichProgressBar(),
        ],
    )
    trainer.fit(model, datamodule=datamodule)
