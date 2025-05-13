import torch
from lightning import Trainer
from lightning.pytorch import callbacks, cli

import datamodules
from models import DiffusionTransformer


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    parser = cli.LightningCLI(
        DiffusionTransformer,
        seed_everything_default=42,
        run=False,
        save_config_callback=None,
    )
    
    trainer = Trainer(
        max_time="00:24:00:00",
        precision="16-mixed",
        gradient_clip_val=1.0,
        callbacks=[
            callbacks.ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                save_weights_only=True,
                save_last=True,
            )
        ],
    )
    trainer.fit(model=parser.model, datamodule=parser.datamodule)