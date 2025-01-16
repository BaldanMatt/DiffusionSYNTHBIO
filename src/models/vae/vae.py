import torch
from torch import nn
from torch.nn import functional as F
import math
import treescope as ts


class Conv1d(nn.Conv1d):
    def forward(self, x):
        x = x.transpose(-1, -2)  # to channnel first convention
        x = super().forward(x)
        x = x.transpose(-1, -2)  # back to channel last convention
        return x


class ResidualBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, activation=nn.SiLU()):
        super().__init__()
        self.residual = nn.Sequential(
            nn.LayerNorm(dim),
            activation,
            Conv1d(dim, dim, kernel_size, padding="same"),
            nn.LayerNorm(dim),
            activation,
            Conv1d(dim, dim, kernel_size, padding="same"),
        )

    def forward(self, x):
        return x + self.residual(x)


class DownSample(nn.Module):
    def __init__(self, in_dim, out_dim, factor=2):
        super().__init__()
        self.kernel = nn.Linear(in_dim * factor, out_dim)
        self.factor = factor

    def forward(self, x):
        *B, L, D = x.shape
        x = x.reshape(*B, L // self.factor, self.factor * D)
        return self.kernel(x)


class UpSample(nn.Module):
    def __init__(self, in_dim, out_dim, factor=2):
        super().__init__()
        self.kernel = nn.Linear(in_dim // factor, out_dim)
        self.factor = factor

    def forward(self, x):
        *B, L, D = x.shape
        x = x.reshape(*B, L * self.factor, D // self.factor)
        return self.kernel(x)


class SequenceBVAE(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=5, activation=nn.SiLU(), beta=1.0):
        super().__init__()
        self.beta = beta
        self.encoder = nn.Sequential(
            Conv1d(in_dim, 8, kernel_size, padding="same"),
            ResidualBlock(8, kernel_size, activation),
            ResidualBlock(8, kernel_size, activation),
            ResidualBlock(8, kernel_size, activation),
            DownSample(8, 16, factor=4),
            ResidualBlock(16, kernel_size, activation),
            ResidualBlock(16, kernel_size, activation),
            ResidualBlock(16, kernel_size, activation),
            DownSample(16, 32, factor=4),
            ResidualBlock(32, kernel_size, activation),
            ResidualBlock(32, kernel_size, activation),
            ResidualBlock(32, kernel_size, activation),
            Conv1d(32, 64, kernel_size, padding="same"),
        )
        self.decoder = nn.Sequential(
            ResidualBlock(32, kernel_size, activation),
            ResidualBlock(32, kernel_size, activation),
            ResidualBlock(32, kernel_size, activation),
            UpSample(32, 16, factor=4),
            ResidualBlock(16, kernel_size, activation),
            ResidualBlock(16, kernel_size, activation),
            ResidualBlock(16, kernel_size, activation),
            UpSample(16, 8, factor=4),
            ResidualBlock(8, kernel_size, activation),
            ResidualBlock(8, kernel_size, activation),
            ResidualBlock(8, kernel_size, activation),
            Conv1d(8, in_dim, kernel_size, padding="same"),
        )

    def encode(self, x):
        mu, sigma = torch.chunk(self.encoder(x), 2, dim=-1)
        return mu, sigma

    def decode(self, z):
        return self.decoder(z)

    def losses(self, x):
        mu, sigma = self.encode(x)
        z = mu + sigma * torch.randn_like(mu)
        x_recon = self.decode(z)
        kl = 0.5 * (sigma**2 + mu**2 - (1e-8 + sigma**2).log() - 1).mean()
        recon = (x_recon - x).pow(2).mean()
        return recon, self.beta * kl
