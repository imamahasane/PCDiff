from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=timesteps.device).float() / half)
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class GroupNorm32(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_ch: int, dropout: float):
        super().__init__()
        self.norm1 = GroupNorm32(32, in_ch)
        self.act1 = SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.emb_proj = nn.Sequential(SiLU(), nn.Linear(emb_ch, out_ch))

        self.norm2 = GroupNorm32(32, out_ch)
        self.act2 = SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.emb_proj(emb)[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)

class AttentionBlock(nn.Module):
    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        if ch % num_heads != 0:
            raise ValueError(f"channels {ch} must be divisible by num_heads {num_heads}")
        self.ch = ch
        self.num_heads = num_heads
        self.norm = GroupNorm32(32, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = torch.chunk(qkv, 3, dim=1)
        head_dim = c // self.num_heads

        q = q.view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)  # b,heads,hw,hd
        k = k.view(b, self.num_heads, head_dim, h * w)  # b,heads,hd,hw
        v = v.view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)  # b,heads,hw,hd

        scale = 1.0 / math.sqrt(head_dim)
        attn = torch.softmax(torch.matmul(q * scale, k), dim=-1)  # b,heads,hw,hw
        h_ = torch.matmul(attn, v)  # b,heads,hw,hd
        h_ = h_.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)
        return x + self.proj(h_)

class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)

class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

class TimestepEmbedSequential(nn.Sequential):

    def __init__(self, *layers: nn.Module, needs_skip: bool = False):
        super().__init__(*layers)
        self.needs_skip = needs_skip

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

@dataclass
class UNetConfig:
    in_channels: int = 2
    out_channels: int = 1
    base_channels: int = 128
    channel_mult: Sequence[int] = (1, 1, 2, 2, 4, 4)
    num_res_blocks: int = 2
    attention_resolutions: Sequence[int] = (16, 8)
    num_heads: int = 4
    dropout: float = 0.0

class UNetModel(nn.Module):
    

    def __init__(self, cfg: UNetConfig, image_size: int):
        super().__init__()
        self.cfg = cfg
        self.image_size = image_size

        time_emb_dim = cfg.base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.base_channels, time_emb_dim),
            SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.in_conv = nn.Conv2d(cfg.in_channels, cfg.base_channels, 3, padding=1)

        ch = cfg.base_channels
        ds = 1
        chs: List[int] = [ch]

        self.input_blocks = nn.ModuleList()
        for level, mult in enumerate(cfg.channel_mult):
            out_ch = cfg.base_channels * mult
            for _ in range(cfg.num_res_blocks):
                layers: List[nn.Module] = [ResBlock(ch, out_ch, time_emb_dim, cfg.dropout)]
                ch = out_ch
                if (image_size // ds) in cfg.attention_resolutions:
                    layers.append(AttentionBlock(ch, cfg.num_heads))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                chs.append(ch)
            if level != len(cfg.channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch)))
                ds *= 2
                chs.append(ch)

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, ch, time_emb_dim, cfg.dropout),
            AttentionBlock(ch, cfg.num_heads),
            ResBlock(ch, ch, time_emb_dim, cfg.dropout),
        )

        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(cfg.channel_mult))[::-1]:
            out_ch = cfg.base_channels * mult
            for _ in range(cfg.num_res_blocks + 1):
                skip_ch = chs.pop()
                layers: List[nn.Module] = [ResBlock(ch + skip_ch, out_ch, time_emb_dim, cfg.dropout)]
                ch = out_ch
                if (image_size // ds) in cfg.attention_resolutions:
                    layers.append(AttentionBlock(ch, cfg.num_heads))
                self.output_blocks.append(TimestepEmbedSequential(*layers, needs_skip=True))
            if level != 0:
                self.output_blocks.append(TimestepEmbedSequential(Upsample(ch), needs_skip=False))
                ds //= 2

        self.out_norm = GroupNorm32(32, ch)
        self.out_act = SiLU()
        self.out_conv = nn.Conv2d(ch, cfg.out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            t = t.view(-1)
        emb = timestep_embedding(t, self.cfg.base_channels)
        emb = self.time_mlp(emb)

        hs: List[torch.Tensor] = []
        h = self.in_conv(x)
        hs.append(h)

        for block in self.input_blocks:
            h = block(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        for block in self.output_blocks:
            if block.needs_skip:
                h = torch.cat([h, hs.pop()], dim=1)
            h = block(h, emb)

        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h
