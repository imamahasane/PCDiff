from __future__ import annotations

import torch
import torch.nn as nn


class _TransformerBlock(nn.Module):
    """Standard pre-norm ViT-style block: MHSA + MLP, applied to a flattened
    spatial feature map."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class _ConvStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Hformer(nn.Module):
   
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 48, num_transformer_blocks: int = 4, num_heads: int = 4):
        super().__init__()
        c = base_channels
        self.stem = _ConvStage(in_channels, c)
        self.down1 = _ConvStage(c, c * 2, stride=2)
        self.down2 = _ConvStage(c * 2, c * 4, stride=2)

        self.token_dim = c * 4
        self.transformer_blocks = nn.ModuleList(
            [_TransformerBlock(self.token_dim, num_heads=num_heads) for _ in range(num_transformer_blocks)]
        )

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = _ConvStage(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = _ConvStage(c * 2, c)

        self.out_conv = nn.Conv2d(c, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)  # (B,c,H,W)
        s1 = self.down1(s0)  # (B,2c,H/2,W/2)
        s2 = self.down2(s1)  # (B,4c,H/4,W/4)

        b, c, h, w = s2.shape
        tokens = s2.flatten(2).transpose(1, 2)  # (B, H*W, 4c)
        for blk in self.transformer_blocks:
            tokens = blk(tokens)
        s2 = tokens.transpose(1, 2).view(b, c, h, w)

        d2 = self.dec2(torch.cat([self.up2(s2), s1], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s0], dim=1))

        return x + self.out_conv(d1)
