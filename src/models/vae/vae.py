import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size, padding="same"),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size, padding="same"),
        )

    def forward(self, x):
        return x + self.residual(x)


class DownSample(nn.Conv1d):
    def __init__(self, dim: int, factor: int = 4):
        super().__init__(dim, dim, kernel_size=factor, stride=factor)


class UpSample(nn.ConvTranspose1d):
    def __init__(self, dim: int, factor: int = 4):
        super().__init__(dim, dim, kernel_size=factor, stride=factor)


def concat_vector_along_sequence(x, c):
    # x: (... L D)  c: (... C) --> result: (... L D+C)
    c = c.unsqueeze(-2).expand(*x.shape[:-1], c.shape[-1])
    return torch.cat([x, c], dim=-1)


class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoded_dim: int,
        hidden_dim: int = 32,
        cond_dim: int = 0,
        stage_depth: int = 4,
        pool: int = 4,
    ):
        super().__init__()
        self.compression = encoded_dim / (4**3)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim + cond_dim, hidden_dim, kernel_size=1),
            *(ResBlock(hidden_dim) for _ in range(stage_depth)),
            DownSample(hidden_dim, factor=pool),
            *(ResBlock(hidden_dim) for _ in range(stage_depth)),
            DownSample(hidden_dim, factor=pool),
            *(ResBlock(hidden_dim) for _ in range(stage_depth)),
            DownSample(hidden_dim, factor=pool),
            *(ResBlock(hidden_dim) for _ in range(stage_depth)),
            nn.Conv1d(hidden_dim, 2 * encoded_dim, kernel_size=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(encoded_dim + cond_dim, hidden_dim, kernel_size=1),
            *(ResBlock(hidden_dim) for _ in range(stage_depth)),
            UpSample(hidden_dim, factor=pool),
            *(ResBlock(hidden_dim) for _ in range(stage_depth)),
            UpSample(hidden_dim, factor=pool),
            *(ResBlock(hidden_dim) for _ in range(stage_depth)),
            UpSample(hidden_dim, factor=pool),
            *(ResBlock(hidden_dim) for _ in range(stage_depth)),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=1),
        )

    def encode(self, x, c=None):
        if c is not None:
            x = concat_vector_along_sequence(x, c)
        x = x.transpose(-1, -2)
        x = self.encoder(x)
        x = x.transpose(-1, -2)

        # split into mu and sigma
        mu, sigma = torch.chunk(x, 2, dim=-1)
        return mu, sigma

    def decode(self, x, c=None):
        if c is not None:
            x = concat_vector_along_sequence(x, c)
        x = x.transpose(-1, -2)
        x = self.decoder(x)
        x = x.transpose(-1, -2)

        # soft clip to +-10 logits
        x = torch.tanh(x / 10) * 10
        return x

    def train_step(self, x, c=None, beta=1.0):
        mu, sigma = self.encode(x, c)
        z = mu + sigma * torch.randn_like(mu)
        x_recon = self.decode(z, c)

        loss_recon = F.cross_entropy(x_recon.transpose(-1, -2), x.transpose(-1, -2))
        loss_kl = 0.5 * (sigma**2 + mu**2 - (1e-8 + sigma**2).log() - 1)
        loss_kl = loss_kl.sum(-1).sum(-1).mean() / (x.shape[-1] * x.shape[-2])
        loss = loss_recon + beta * loss_kl
        return loss, {
            "loss_recon": loss_recon.item(),
            "loss_kl": loss_kl.item(),
            "beta": beta,
        }


class VQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoded_dim: int,
        hidden_dim: int = 32,
        cond_dim: int = 0,
        fsq_levels: int = 2,
        stage_depth: int = 4,
        pool: int = 4,
    ):
        super().__init__()
        self.fsq_levels = fsq_levels
        self.compression = fsq_levels / input_dim * (encoded_dim / (4**3))
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim + cond_dim, hidden_dim, kernel_size=1),
            *(ResBlock(hidden_dim) for _ in range(stage_depth)),
            DownSample(hidden_dim, factor=pool),
            *(ResBlock(hidden_dim) for _ in range(stage_depth)),
            DownSample(hidden_dim, factor=pool),
            *(ResBlock(hidden_dim) for _ in range(stage_depth)),
            DownSample(hidden_dim, factor=pool),
            *(ResBlock(hidden_dim) for _ in range(stage_depth)),
            nn.Conv1d(hidden_dim, encoded_dim, kernel_size=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(encoded_dim + cond_dim, hidden_dim, kernel_size=1),
            *(ResBlock(hidden_dim) for _ in range(stage_depth)),
            UpSample(hidden_dim, factor=pool),
            *(ResBlock(hidden_dim) for _ in range(stage_depth)),
            UpSample(hidden_dim, factor=pool),
            *(ResBlock(hidden_dim) for _ in range(stage_depth)),
            UpSample(hidden_dim, factor=pool),
            *(ResBlock(hidden_dim) for _ in range(stage_depth)),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=1),
        )

    def encode(self, x, c=None):
        if c is not None:
            x = concat_vector_along_sequence(x, c)
        x = x.transpose(-1, -2)
        x = self.encoder(x)
        x = x.transpose(-1, -2)

        # quantization step
        x = self.fsq_levels * torch.sigmoid(x)  # bound z to (0, L)
        x = x + (x.floor() - x).detach()  # discretize with ste
        x = x - (self.fsq_levels - 1) / 2  # recenter to (-L/2, L/2)
        return x

    def decode(self, x, c=None):
        if c is not None:
            x = concat_vector_along_sequence(x, c)
        x = x.transpose(-1, -2)
        x = self.decoder(x)
        x = x.transpose(-1, -2)

        # soft clip to +-10 logits
        x = torch.tanh(x / 10) * 10
        return x

    def train_step(self, x, c=None):
        z = self.encode(x, c)
        x_recon = self.decode(z, c)

        loss = F.cross_entropy(x_recon.transpose(-1, -2), x.transpose(-1, -2))
        return loss, {"loss_recon": loss.item()}
