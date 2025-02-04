import torch
from torch import nn
from torch.nn import functional as F
from lightning import LightningModule
from einops import rearrange


class FeedForward(nn.Sequential):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        qkv = rearrange(self.qkv(x), "B N (H D) -> B H N D", H=self.num_heads)
        q, k, v = qkv.chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "B H N D -> B N (H D)")
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, expand: int = 4):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = FeedForward(dim, dim * expand, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SinusoidalEmbed(nn.Module):
    def __init__(self, embed_dim: int, period: float = 1.0, n_freqs: int = 128):
        super().__init__()
        freqs = torch.exp(
            -torch.log(torch.tensor(period)) * torch.linspace(0, 1, n_freqs)
        )
        self.register_buffer("freqs", freqs)
        self.feedforward = FeedForward(2 * n_freqs, embed_dim, embed_dim)

    def forward(self, t):
        angles = self.freqs * t.unsqueeze(-1)
        x = torch.cat([angles.sin(), angles.cos()], dim=-1)
        x = self.feedforward(x)
        return x


class DiffusionTransformer(LightningModule):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        hidden_dim: int = 128,
        depth: int = 8,
        num_heads: int = 8,
        patch_size: int = 4,
        *,
        x_jitter_std: float = 0.01,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        # embedding layers
        self.x_embed = FeedForward(input_dim * patch_size, hidden_dim, hidden_dim)
        self.x_unembed = FeedForward(hidden_dim, hidden_dim, input_dim * patch_size)
        self.c_embed = FeedForward(cond_dim, hidden_dim, hidden_dim)
        self.pos_embed = SinusoidalEmbed(hidden_dim)
        self.time_embed = SinusoidalEmbed(hidden_dim)

        # transformer layers
        self.blocks = nn.Sequential(
            *[Block(hidden_dim, num_heads) for _ in range(depth)]
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )

    def forward(self, x, t, y):
        # patchify and embed
        x = rearrange(x, "B (L P) D -> B L (P D)", P=self.hparams["patch_size"])
        pos = torch.linspace(0, 1, x.shape[-2], device=x.device, dtype=x.dtype)
        x = self.x_embed(x) + self.pos_embed(pos)

        # add conditioning
        c = self.c_embed(y) + self.time_embed(t)
        x = x + c.unsqueeze(-2)

        # transformer blocks
        x = self.blocks(x)

        # unembed and unpatchify
        x = self.x_unembed(x)
        x = rearrange(x, "B L (P D) -> B (L P) D", P=self.hparams["patch_size"])
        return x

    def push(self, x, y, n_steps=16):
        dt = 1.0 / n_steps
        t = torch.zeros_like(y[..., 0])
        for _ in range(n_steps):
            # integration with runge-kutta
            k1 = self(x, t, y)
            k2 = self(x + k1 * dt / 2, t + dt / 2, y)
            k3 = self(x + k2 * dt / 2, t + dt / 2, y)
            k4 = self(x + k3 * dt, t + dt, y)
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
            t = t + dt
        return x

    def training_step(self, batch, batch_idx):
        (x1, y) = batch
        t = torch.rand_like(y[..., 0])
        x0 = torch.rand_like(x1)

        xt = x1 * t[..., None, None] + x0 * (1 - t[..., None, None])
        xt = xt + torch.randn_like(xt) * self.hparams["x_jitter_std"]

        loss = F.mse_loss(self(xt, t, y), x1 - x0)
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            (x, y) = batch
            classes = y.shape[-1]
            # generate 1 sample for each class
            y = torch.eye(classes, device=y.device, dtype=y.dtype)
            x0 = torch.rand_like(x)[:classes]
            x1 = self.push(x0, y)
            self.logger.log_image("generated", [el.T.unsqueeze(0) for el in x1])
            self.logger.log_image("sampled", [el.T.unsqueeze(0) for el in x[:classes]])

    def test_step(self, batch, batch_idx):
        # TODO: implement test step
        print("Skipping test step")
        pass
