import math
import torch
from torch import nn
from torch.nn import functional as F
from lightning import LightningModule
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: nn.Module = nn.SiLU(),
        zero_init: bool = False,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.activation = activation
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.activation(x)
        x = self.out_proj(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        qkv = rearrange(self.qkv_proj(x), "B N (H D) -> B H N D", H=self.num_heads)
        q, k, v = qkv.chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "B H N D -> B N (H D)")
        x = self.out_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, expand: int = 4):
        super().__init__()
        self.modulation = FeedForward(dim, dim, 6 * dim, zero_init=True)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, bias=False)
        self.attention = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, bias=False)
        self.feedforward = FeedForward(dim, dim * expand, dim)

    def forward(self, x, c):
        c = self.modulation(c)
        shift1, scale1, gate1, shift2, scale2, gate2 = c.chunk(6, dim=-1)
        x = x + gate1 * self.attention(shift1 + (1 + scale1) * self.norm1(x))
        x = x + gate2 * self.feedforward(shift2 + (1 + scale2) * self.norm2(x))
        return x


class SinusoidalEmbed(nn.Module):
    def __init__(self, dim: int, period: float = 2 * math.pi, n_freqs: int = 256):
        super().__init__()
        freqs = torch.exp(-math.log(period) * torch.linspace(0, 1, n_freqs))
        self.register_buffer("freqs", freqs)
        self.feedforward = FeedForward(2 * n_freqs, dim, dim)
        self.freqs: torch.Tensor

    def forward(self, t):
        angles = 2 * math.pi * self.freqs * t
        x = torch.cat([angles.sin(), angles.cos()], dim=-1)
        x = self.feedforward(x)
        return x


class DiffusionTransformer(LightningModule):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        hidden_dim: int,
        num_heads: int,
        depth: int,
        patch_size: int = 1,
        *,
        jitter_std: float = 0.01,
        drop_cond_rate: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.c_embed = FeedForward(cond_dim, hidden_dim, hidden_dim)
        self.time_embed = SinusoidalEmbed(hidden_dim)
        self.pos_embed = SinusoidalEmbed(hidden_dim)
        self.x_embed = nn.Linear(input_dim * patch_size, hidden_dim)
        self.x_unembed = nn.Linear(hidden_dim, input_dim * patch_size)
        self.blocks = nn.ModuleList(Block(hidden_dim, num_heads) for _ in range(depth))

    def configure_optimizers(self):
        lr = self.hparams["learning_rate"]
        wd = self.hparams["weight_decay"]
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

    def forward(self, x, t, y=None):
        # condition embed
        c = self.time_embed(t)
        if y is not None:
            c += self.c_embed(y)
        c = c.unsqueeze(-2)  # shape: B C -> B 1 C

        # patch embed
        x = rearrange(x, "B (L P) D -> B L (P D)", P=self.hparams["patch_size"])
        pos = torch.linspace(0, 1, x.shape[-2], device=x.device, dtype=x.dtype)
        x = self.x_embed(x) + self.pos_embed(pos.unsqueeze(-1))

        # transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # patch unembed
        x = self.x_unembed(x)
        x = rearrange(x, "B L (P D) -> B (L P) D", P=self.hparams["patch_size"])
        return x

    def push(self, x, y=None, guidance=1.0, n_steps=16):
        def flow(x, t, y):
            return guidance * self(x, t, y) + (1.0 - guidance) * self(x, t, y=None)

        dt = 1.0 / n_steps
        *B, L, D = x.shape
        t = torch.zeros(*B, 1, device=x.device)
        for _ in range(n_steps):
            # integration with runge-kutta
            k1 = flow(x, t, y)
            k2 = flow(x + k1 * dt / 2, t + dt / 2, y)
            k3 = flow(x + k2 * dt / 2, t + dt / 2, y)
            k4 = flow(x + k3 * dt, t + dt, y)
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
            t = t + dt
        return x

    def loss(self, x1, y=None):
        *B, L, D = x1.shape
        t = torch.sigmoid(torch.randn(*B, 1, device=x1.device))
        x0 = torch.randn(*B, L, D, device=x1.device)

        xt = x1 * t[..., None] + x0 * (1 - t[..., None])
        xt += self.hparams["jitter_std"] * torch.randn_like(xt)

        target = x1 - x0
        flow = self(xt, t, y)
        return F.mse_loss(flow, target)

    def training_step(self, batch, batch_idx):
        (x1, y) = batch
        loss_conditional = self.loss(x1, y)
        loss_unconditional = self.loss(x1, y=None)
        loss = loss_conditional + self.hparams["drop_cond_rate"] * loss_unconditional
        self.log("train/loss_conditional", loss_conditional)
        self.log("train/loss_unconditional", loss_unconditional)
        self.log("train/loss_total", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (x1, y) = batch
        loss_conditional = self.loss(x1, y)
        loss_unconditional = self.loss(x1, y=None)
        loss = loss_conditional + self.hparams["drop_cond_rate"] * loss_unconditional
        self.log("val/loss_conditional", loss_conditional, on_epoch=True)
        self.log("val/loss_unconditional", loss_unconditional, on_epoch=True)
        self.log("val/loss_total", loss, prog_bar=True, on_epoch=True)
