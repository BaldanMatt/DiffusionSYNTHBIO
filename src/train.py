from lightning import Trainer
from lightning.pytorch import callbacks, loggers, cli

from datamodules import DiscDiffDataModule, DHSDataModule
from models import DiffusionTransformer


if __name__ == "__main__":
    parser = cli.LightningCLI(
        DiffusionTransformer,
        seed_everything_default=42,
        run=False,
        save_config_callback=None,
    )
    logger = loggers.WandbLogger(project="DNAdiffusion", log_model=True)
    trainer = Trainer(
        max_time="00:24:00:00",
        precision="16-mixed",
        logger=logger,
        gradient_clip_val=1.0,
        callbacks=[
            callbacks.ModelCheckpoint(
                monitor="val/loss_conditional",
                mode="min",
                save_weights_only=True,
                save_last=True,
            ),
            callbacks.EarlyStopping(
                monitor="val/loss_conditional",
                mode="min",
                patience=10,
            ),
        ],
    )
    trainer.fit(model=parser.model, datamodule=parser.datamodule)
