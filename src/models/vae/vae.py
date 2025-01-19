import torch
from torch import nn
from torch.nn import functional as F


class Conv1d(nn.Conv1d):
    def forward(self, x):
        x = x.transpose(-1, -2)  # to channnel first convention
        x = super().forward(x)
        x = x.transpose(-1, -2)  # back to channel last convention
        return x


class ResidualBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, stride=1, padding="same", dilation=1):
        assert padding == "same", "Only 'same' padding is supported"
        assert stride == 1, "Only stride=1 is supported"
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim))
        self.residual = nn.Sequential(
            nn.RMSNorm(dim, elementwise_affine=False),
            nn.SiLU(inplace=True),
            Conv1d(dim, dim, kernel_size, stride, padding, dilation),
            nn.RMSNorm(dim, elementwise_affine=False),
            nn.SiLU(inplace=True),
            Conv1d(dim, dim, kernel_size, stride, padding, dilation),
        )

    def forward(self, x):
        return x + self.gate * self.residual(x)


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


class SequenceVAE(nn.Module):
    def __init__(self, in_dim, dim=32, depth=3, blocks=4, kernel_size=5):
        super().__init__()
        assert depth == 3, "Only depth=3 implemented"
        self.compression = (dim / in_dim) / (4**depth)
        self.encoder = nn.Sequential(
            Conv1d(in_dim, dim, kernel_size, padding="same"),
            *(ResidualBlock(dim, kernel_size) for _ in range(blocks)),
            DownSample(dim, dim, factor=4),
            *(ResidualBlock(dim, kernel_size) for _ in range(blocks)),
            DownSample(dim, dim, factor=4),
            *(ResidualBlock(dim, kernel_size) for _ in range(blocks)),
            DownSample(dim, dim, factor=4),
            *(ResidualBlock(dim, kernel_size) for _ in range(blocks)),
            Conv1d(dim, 2 * dim, kernel_size, padding="same"),
        )
        self.decoder = nn.Sequential(
            *(ResidualBlock(dim, kernel_size) for _ in range(blocks)),
            UpSample(dim, dim, factor=4),
            *(ResidualBlock(dim, kernel_size) for _ in range(blocks)),
            UpSample(dim, dim, factor=4),
            *(ResidualBlock(dim, kernel_size) for _ in range(blocks)),
            UpSample(dim, dim, factor=4),
            *(ResidualBlock(dim, kernel_size) for _ in range(blocks)),
            Conv1d(dim, in_dim, kernel_size, padding="same"),
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

        recon = F.cross_entropy(x_recon.transpose(-1, -2), x.transpose(-1, -2))
        kl = 0.5 * (sigma**2 + mu**2 - (1e-8 + sigma**2).log() - 1)
        kl = kl.sum(-1).sum(-1).mean() / (x.shape[-1] * x.shape[-2])
        return recon, kl
