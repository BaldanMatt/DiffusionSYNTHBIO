import torch
from torch import nn, Tensor


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


class Encoder(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int, blocks: int = 4):
        super().__init__(
            nn.Conv1d(input_dim, 16, kernel_size=1),  # inplace conv
            *(Block(16) for _ in range(blocks)),
            nn.Conv1d(16, 32, kernel_size=4, stride=4),  # downsample x4
            *(Block(32) for _ in range(blocks)),
            nn.Conv1d(32, 64, kernel_size=4, stride=4),  # downsample x4
            *(Block(64) for _ in range(blocks)),
            nn.Conv1d(64, 128, kernel_size=4, stride=4),  # downsample x4
            *(Block(128) for _ in range(blocks)),
            nn.Conv1d(128, output_dim, kernel_size=1),  # inplace conv
        )


class Decoder(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int, blocks: int = 4):
        super().__init__(
            nn.Conv1d(input_dim, 128, kernel_size=1),  # inplace conv
            *(Block(128) for _ in range(blocks)),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=4),  # upsample x4
            *(Block(64) for _ in range(blocks)),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=4),  # upsample x4
            *(Block(32) for _ in range(blocks)),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=4),  # upsample x4
            *(Block(16) for _ in range(blocks)),
            nn.Conv1d(16, output_dim, kernel_size=1),  # inplace conv
        )


class VAE(nn.Module):
    def __init__(self, input_dim: int, encoded_dim: int, blocks: int = 4):
        super().__init__()
        self.compression = encoded_dim / (4**3)
        self.encoder = Encoder(input_dim, 2 * encoded_dim, blocks)
        self.decoder = Decoder(encoded_dim, input_dim, blocks)

    def encode(self, x: Tensor):
        x = self.encoder(x)
        mu, sigma = torch.chunk(x, 2, dim=-2)
        sigma = sigma.abs() + 1e-8
        return mu, sigma

    def decode(self, x: Tensor):
        x = self.decoder(x)
        return x


class VQVAE(VAE):
    def __init__(
        self,
        input_dim: int,
        encoded_dim: int,
        blocks: int = 4,
        fsq_levels: int = 3,
    ):
        super().__init__(input_dim, encoded_dim, blocks)
        self.compression *= fsq_levels / input_dim
        self.fsq_levels = fsq_levels

    def encode(self, x: Tensor):
        z, sigma = super().encode(x)
        z = self.fsq_levels * torch.sigmoid(z)  # bound z to (0, L)
        z = z + (z.floor() - z).detach()  # discretize with ste
        z = z - (self.fsq_levels - 1) / 2  # recenter to (-L/2, L/2)
        return z
