import torch
from torch import nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, *, zero_init=False):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        if zero_init:
            nn.init.zeros_(self.lin2.weight)
            nn.init.zeros_(self.lin2.bias)

    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))


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


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, expand: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, dim * expand, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        embed_dim: int,
        sequence_len: int = 2048,
        patch_size: int = 8,
    ):
        super().__init__()
        self.patch = patch_size
        self.pos_embed = nn.Embedding(sequence_len // patch_size, embed_dim)
        self.patch_embed = nn.Linear(input_dim * patch_size, embed_dim)
        self.cond_embed = nn.Linear(cond_dim, embed_dim)
        self.encoder = FeedForward(embed_dim, embed_dim, embed_dim)
        self.decoder = FeedForward(embed_dim, embed_dim, input_dim * patch_size)

    def patchify(self, x):
        *B, L, D = x.shape
        return x.view(*B, L // self.patch, self.patch * D)

    def unpatchify(self, x):
        *B, L, D = x.shape
        return x.view(*B, L * self.patch, D // self.patch)

    def encode(self, x, c):
        if c.ndim < x.ndim:
            c = c.unsqueeze(-2)
        x = self.patchify(x)
        pos = torch.arange(x.shape[-2], device=x.device)
        x = self.encoder(self.pos_embed(pos) + self.patch_embed(x) + self.cond_embed(c))
        return x

    def decode(self, x):
        x = self.decoder(x)
        x = self.unpatchify(x)
        return x


class DiffusionTransformer(nn.Module):
    def __init__(
        self, input_dim: int, cond_dim: int, hidden_dim: int, depth: int, num_heads: int
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(input_dim, cond_dim + 1, hidden_dim)
        self.blocks = nn.Sequential(
            *[Block(hidden_dim, num_heads) for _ in range(depth)]
        )

    def forward(self, x, t, y):
        c = torch.cat([t.unsqueeze(-1), y], dim=-1)
        x = self.patch_embed.encode(x, c)
        x = self.blocks(x)
        x = self.patch_embed.decode(x)
        return x
