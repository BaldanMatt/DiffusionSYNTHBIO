from lightning import Trainer
from lightning.pytorch import callbacks, loggers

from data import TestDataModule
from vae import BetaVAE


config: dict = dict(
    input_dim=5,
    encoded_dim=64,
    hidden_dim=64,
    blocks=4,
    beta_max=1.0,
    cycle_steps=10000,
    learning_rate=3e-4,
    weight_decay=1e-3,
)


if __name__ == "__main__":
    datamodule = TestDataModule(
        "hf://datasets/Zehui127127/latent-dna-diffusion/sequence.csv", batch_size=256
    )
    model = BetaVAE(**config)

    logger = loggers.WandbLogger(project="DNAdiffusion", log_model=True)
    trainer = Trainer(
        max_epochs=100,
        logger=logger,
        gradient_clip_val=0.5,
        callbacks=[
            callbacks.ModelCheckpoint(every_n_epochs=10),
            callbacks.RichModelSummary(),
            callbacks.RichProgressBar(),
        ],
    )
    trainer.fit(model, datamodule=datamodule)
