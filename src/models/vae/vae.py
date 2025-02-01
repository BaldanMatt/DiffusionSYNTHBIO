from typing import Any
import torch
from torch import nn, Tensor
from lightning import LightningModule


class Block(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding="same"),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size, padding="same"),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor):
        return x + self.layers(x)


class DownSample(nn.Conv1d):
    def __init__(self, input_dim: int, output_dim: int, factor: int = 4):
        super().__init__(input_dim, output_dim, kernel_size=factor, stride=factor)


class UpSample(nn.ConvTranspose1d):
    def __init__(self, input_dim: int, output_dim: int, factor: int = 4):
        super().__init__(input_dim, output_dim, kernel_size=factor, stride=factor)


class Encoder(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        blocks: int = 4,
    ):
        super().__init__(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            *(Block(hidden_dim) for _ in range(blocks)),
            DownSample(hidden_dim, hidden_dim, factor=4),
            *(Block(hidden_dim) for _ in range(blocks)),
            DownSample(hidden_dim, hidden_dim, factor=4),
            *(Block(hidden_dim) for _ in range(blocks)),
            DownSample(hidden_dim, hidden_dim, factor=4),
            *(Block(hidden_dim) for _ in range(blocks)),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1),
        )


class Decoder(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        blocks: int = 4,
    ):
        super().__init__(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            *(Block(hidden_dim) for _ in range(blocks)),
            UpSample(hidden_dim, hidden_dim, factor=4),
            *(Block(hidden_dim) for _ in range(blocks)),
            UpSample(hidden_dim, hidden_dim, factor=4),
            *(Block(hidden_dim) for _ in range(blocks)),
            UpSample(hidden_dim, hidden_dim, factor=4),
            *(Block(hidden_dim) for _ in range(blocks)),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1),
        )


class BetaVAE(LightningModule):
    def __init__(
        self,
        input_dim: int,
        encoded_dim: int,
        hidden_dim: int = 64,
        blocks: int = 4,
        *,
        beta_max: float = 1.0,
        cycle_steps: int = 1,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.step = 0
        self.compression = encoded_dim / (4**3)
        self.encoder = Encoder(input_dim, 2 * encoded_dim, hidden_dim, blocks)
        self.decoder = Decoder(encoded_dim, input_dim, hidden_dim, blocks)

    def encode(self, x: Tensor):
        x = self.encoder(x)
        mu, sigma = torch.chunk(x, 2, dim=-2)
        sigma = sigma.abs() + 1e-8
        return mu, sigma

    def decode(self, x: Tensor):
        x = self.decoder(x)
        return x

    def configure_optimizers(self):
        lr = self.hparams["learning_rate"]
        wd = self.hparams["weight_decay"]
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)

    def training_step(self, batch, batch_idx):
        (x,) = batch
        x = x.transpose(-1, -2)

        # beta scheduling
        cycle_steps = self.hparams["cycle_steps"]
        beta_max = self.hparams["beta_max"]
        self.step = (self.step + 1) % cycle_steps
        beta = beta_max * min(1.0, 2 * self.step / cycle_steps)

        # forward pass
        mu, sigma = self.encode(x)
        z = mu + sigma * torch.randn_like(mu)
        x_recon = self.decode(z)
        x_recon = torch.tanh(x_recon / 10) * 10  # soft clip to (-10, 10)

        # loss
        loss_recon = nn.functional.cross_entropy(x_recon, x)
        factor = 0.5 * (mu.shape[-1] * mu.shape[-2]) / x.shape[-1]
        loss_kl = factor * (sigma**2 + mu**2 - (sigma**2).log() - 1).mean()
        loss = loss_recon + beta * loss_kl
        self.log_dict(
            {"loss_recon": loss_recon, "loss_kl": loss_kl, "beta": beta, "elbo": loss},
            prog_bar=True,
        )
        return loss


class VQVAE(LightningModule):
    def __init__(
        self,
        input_dim: int,
        encoded_dim: int,
        hidden_dim: int = 64,
        blocks: int = 4,
        fsq_levels: int = 3,
        *,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.compression = encoded_dim * (fsq_levels / input_dim) / (4**3)
        self.encoder = Encoder(input_dim, encoded_dim, hidden_dim, blocks)
        self.decoder = Decoder(encoded_dim, input_dim, hidden_dim, blocks)

    def encode(self, x: Tensor):
        x = self.encoder(x)
        return x

    def decode(self, x: Tensor):
        levels = self.hparams["fsq_levels"]
        z = self.decoder(x)
        z = levels * torch.sigmoid(z)  # bound z to (0, L)
        z = z + (z.floor() - z).detach()  # discretize with ste
        z = z - (levels - 1) / 2  # recenter to (-L/2, L/2)
        return z

    def configure_optimizers(self):
        lr = self.hparams["learning_rate"]
        wd = self.hparams["weight_decay"]
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)

    def training_step(self, batch, batch_idx):
        (x,) = batch
        x = x.transpose(-1, -2)

        # forward pass
        z = self.encode(x)
        x_recon = self.decode(z)
        x_recon = torch.tanh(x_recon / 10) * 10  # soft clip to (-10, 10)

        # loss
        loss_recon = nn.functional.cross_entropy(x_recon, x)
        self.log_dict({"loss_recon": loss_recon}, prog_bar=True)
        return loss_recon
