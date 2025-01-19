import torch
from torch import nn
from torch.nn import functional as F


class ConditionalSequential(nn.Sequential):
    def forward(self, x, *, c=None):
        for module in self:
            x = module(x, c=c)
        return x


class ZeroInitLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)


class Conv1d(nn.Conv1d):
    def forward(self, x, *, c=None):
        x = x.transpose(-1, -2)  # to channnel first convention
        x = super().forward(x)
        x = x.transpose(-1, -2)  # back to channel last convention
        return x


class ResidualBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, stride=1, padding="same", dilation=1):
        assert padding == "same", "Only 'same' padding is supported"
        assert stride == 1, "Only stride=1 is supported"
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            ZeroInitLinear(dim, 3 * dim),
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(dim, elementwise_affine=False),
            nn.SiLU(),
            Conv1d(dim, dim, kernel_size, stride, padding, dilation),
            nn.LayerNorm(dim, elementwise_affine=False),
            nn.SiLU(),
            Conv1d(dim, dim, kernel_size, stride, padding, dilation),
        )

    def forward(self, x, *, c):
        c = self.modulation(c.unsqueeze(-2))
        shift, scale, gate = torch.chunk(c, 3, dim=-1)
        x = x + gate * self.residual(shift + (1 + scale) * x)
        return x


class DownSample(nn.Module):
    def __init__(self, in_dim, out_dim, factor=2):
        super().__init__()
        self.kernel = nn.Linear(in_dim * factor, out_dim)
        self.factor = factor

    def forward(self, x, *, c=None):
        *B, L, D = x.shape
        x = x.reshape(*B, L // self.factor, self.factor * D)
        return self.kernel(x)


class UpSample(nn.Module):
    def __init__(self, in_dim, out_dim, factor=2):
        super().__init__()
        self.kernel = nn.Linear(in_dim // factor, out_dim)
        self.factor = factor

    def forward(self, x, *, c=None):
        *B, L, D = x.shape
        x = x.reshape(*B, L * self.factor, D // self.factor)
        return self.kernel(x)


class SequenceCVAE(nn.Module):
    def __init__(self, in_dim, c_dim, dim=32, depth=3, blocks=4, kernel_size=5):
        super().__init__()
        assert depth == 3, "Only depth=3 implemented"
        self.compression = (dim / in_dim) / (4**depth)
        self.conditioning = nn.Sequential(
            nn.Linear(c_dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, dim)
        )
        self.encoder = ConditionalSequential(
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
        self.decoder = ConditionalSequential(
            *(ResidualBlock(dim, kernel_size) for _ in range(blocks)),
            UpSample(dim, dim, factor=4),
            *(ResidualBlock(dim, kernel_size) for _ in range(blocks)),
            UpSample(dim, dim, factor=4),
            *(ResidualBlock(dim, kernel_size) for _ in range(blocks)),
            UpSample(dim, dim, factor=4),
            *(ResidualBlock(dim, kernel_size) for _ in range(blocks)),
            Conv1d(dim, in_dim, kernel_size, padding="same"),
        )

    def encode(self, x, c):
        c = self.conditioning(c)
        x = self.encoder(x, c=c)
        mu, sigma = torch.chunk(x, 2, dim=-1)
        return mu, sigma

    def decode(self, z, c):
        c = self.conditioning(c)
        x = self.decoder(z, c=c)
        return torch.tanh(x / 10) * 10  # soft clip

    def losses(self, x, c):
        mu, sigma = self.encode(x, c=c)
        z = mu + sigma * torch.randn_like(mu)
        x_recon = self.decode(z, c=c)

        recon = F.cross_entropy(x_recon.transpose(-1, -2), x.transpose(-1, -2))
        kl = 0.5 * (sigma**2 + mu**2 - (1e-8 + sigma**2).log() - 1)
        kl = kl.sum(-1).sum(-1).mean() / (x.shape[-1] * x.shape[-2])
        return recon, kl
