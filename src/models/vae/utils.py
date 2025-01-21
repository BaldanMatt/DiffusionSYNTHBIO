import torch
from torch import nn
from torch.nn import functional as F


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


class ConvResidualBlock(nn.Module):
    def __init__(self, dim, kernel_size=5):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, kernel_size, padding="same"),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, kernel_size, padding="same"),
        )

    def forward(self, x):
        x = x.transpose(-1, -2)  # Conv and BN expect channels first
        x = x + self.residual(x)
        return x.transpose(-1, -2)  # Transpose back to channels last


class Linear(nn.Linear):
    def __init__(self, *args, zero_init=False, **kwargs):
        super().__init__(*args, **kwargs)
        if zero_init:
            nn.init.zeros_(self.weight)
            nn.init.zeros_(self.bias)


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
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x