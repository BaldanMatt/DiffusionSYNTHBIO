import torch
from torch import nn
from torch.nn import functional as F


class ZeroInitLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, expand=2):
        super().__init__()
        self.modulation = FeedForward(dim, dim * expand, 6 * dim)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff = FeedForward(dim, dim * expand, dim)

    def forward(self, x, c):
        c = torch.chunk(self.modulation(c), 6, dim=-1)
        shift1, scale1, gate1, shift2, scale2, gate2 = c
        x = x + gate1 * self.attn(shift1 + (1 + scale1) * self.norm1(x))
        x = x + gate2 * self.ff(shift2 + (1 + scale2) * self.norm2(x))
        return x


class PosEmbedding(nn.Module):
    def __init__(self, emb_dim, max_len=256):
        super().__init__()
        self.embed = nn.Embedding(max_len, emb_dim)
        self.ff = FeedForward(emb_dim, 2 * emb_dim, emb_dim)

    def forward(self, x):
        pos = torch.arange(x.size(-2), device=x.device)
        return self.ff(self.embed(pos))


class Patchify(nn.Module):
    def __init__(self, in_dim, emb_dim, patch_size=4, expand=2):
        super().__init__()
        self.ff = FeedForward(in_dim * patch_size, expand * emb_dim, emb_dim)
        self.patch_size = patch_size

    def forward(self, x):
        *B, L, D = x.shape
        x = x.reshape(*B, L // self.patch_size, self.patch_size * D)
        return self.ff(x)


class UnPatchify(nn.Module):
    def __init__(self, in_dim, emb_dim, patch_size=4, expand=2):
        super().__init__()
        self.ff = FeedForward(emb_dim, expand * emb_dim, in_dim * patch_size)
        self.patch_size = patch_size

    def forward(self, x):
        x = self.ff(x)
        *B, L, D = x.shape
        x = x.reshape(*B, L * self.patch_size, D // self.patch_size)
        return x


class SequenceCVQVAE(nn.Module):
    def __init__(
        self, in_dim, c_dim, fsq_maxlevel=2, dim=64, num_heads=4, depth=2, expand=2
    ):
        super().__init__()
        assert depth == 4, "Only depth=4 implemented"
        self.fsq_maxlevel = fsq_maxlevel
        self.conditioning = FeedForward(c_dim, 2 * dim, dim)

        self.patchify = Patchify(in_dim, dim, patch_size=4, expand=expand)
        self.unpatchify = UnPatchify(in_dim, dim, patch_size=4, expand=expand)

        self.encoder_blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads, expand) for _ in range(depth)]
        )
        self.decoder_blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads, expand) for _ in range(depth)]
        )

    def encode(self, x, c):
        c = self.conditioning(c)
        x = self.patchify(x)
        for block in self.encoder_blocks:
            x = block(x, c)
        z = self.fsq_maxlevel * torch.tanh(x)  # bound z to [-L, L]
        return z + (z.round() - z).detach()  # round  and ste

    def decode(self, x, c):
        c = self.conditioning(c)
        for block in self.decoder_blocks:
            x = block(x, c)
        x = self.unpatchify(x)
        return 10 * torch.tanh(x)  # soft clip to +-10 logits

    def losses(self, x, c):
        z = self.encode(x, c=c)
        x_recon = self.decode(z, c=c)
        recon = F.cross_entropy(x_recon.transpose(-1, -2), x.transpose(-1, -2))
        return recon
