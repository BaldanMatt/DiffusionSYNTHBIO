import torch
from torch import nn, Tensor


class Reshape(nn.Module):
    def __init__(self, dim: int, mode: str = "none", factor: int = 2):
        super().__init__()
        self.kernel = {
            "downsample": nn.Conv1d(dim, dim, factor, factor),
            "upsample": nn.ConvTranspose1d(dim, dim, factor, factor),
            "none": nn.Identity(),
        }[mode.lower()]

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(-1, -2)
        x = self.kernel(x)
        x = x.transpose(-1, -2)
        return x


class ConvBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5, activation=nn.SiLU()):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.Linear(dim, dim),
            activation,
            nn.Linear(dim, 3 * dim),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, bias=False)
        self.residual = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding="same"),
            activation,
            nn.Conv1d(dim, dim, kernel_size, padding="same"),
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        c = self.modulation(c)
        shift, scale, gate = c.chunk(3, dim=-1)
        h = shift + self.norm(x) * (1 + scale)

        h = h.transpose(-1, -2)
        h = self.residual(h)
        h = h.transpose(-1, -2)
        return x + gate * h


class ConvNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        stages: int = 6,
        blocks: int = 2,
        reshape: str = "none",
    ):
        super().__init__()
        self.cond_embed = nn.Sequential(nn.Linear(cond_dim, hidden_dim), nn.SiLU())

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(stages):
            self.layers.extend(ConvBlock(hidden_dim) for _ in range(blocks))
            self.layers.append(Reshape(hidden_dim, mode=reshape))
        self.layers.extend(ConvBlock(hidden_dim) for _ in range(blocks))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        c = self.cond_embed(c).unsqueeze(-2)
        for layer in self.layers:
            x = layer(x) if not isinstance(layer, ConvBlock) else layer(x, c)
        return x


class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        encoded_dim: int,
        hidden_dim: int = 64,
        stages: int = 6,
        blocks: int = 2,
    ):
        super().__init__()
        self.compression = encoded_dim / (2**stages)
        self.encoder = ConvNet(
            input_dim,
            cond_dim,
            2 * encoded_dim,
            hidden_dim,
            stages,
            blocks,
            reshape="downsample",
        )
        self.decoder = ConvNet(
            encoded_dim,
            cond_dim,
            input_dim,
            hidden_dim,
            stages,
            blocks,
            reshape="upsample",
        )

    def encode(self, x: Tensor, c: Tensor):
        x = self.encoder(x, c)
        mu, sigma = torch.chunk(x, 2, dim=-1)
        sigma = sigma.abs() + 1e-8
        return mu, sigma

    def decode(self, x: Tensor, c: Tensor):
        return self.decoder(x, c)


class VQVAE(VAE):
    def __init__(self, input_dim: int, *args, fsq_levels: int = 3, **kwargs):
        super().__init__(input_dim, *args, **kwargs)
        self.compression *= fsq_levels / input_dim
        self.fsq_levels = fsq_levels

    def encode(self, x: Tensor, c: Tensor):
        z, sigma = super().encode(x, c)
        z = self.fsq_levels * torch.sigmoid(z)  # bound z to (0, L)
        z = z + (z.floor() - z).detach()  # discretize with ste
        z = z - (self.fsq_levels - 1) / 2  # recenter to (-L/2, L/2)
        return z
