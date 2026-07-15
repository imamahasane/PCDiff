from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ASCONDenoiser(nn.Module):
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 64, embed_dim: int = 128):
        super().__init__()
        c = base_channels
        self.enc1 = _ConvBlock(in_channels, c)
        self.enc2 = _ConvBlock(c, c * 2)
        self.enc3 = _ConvBlock(c * 2, c * 4)
        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = _ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = _ConvBlock(c * 2, c)
        self.out_conv = nn.Conv2d(c, out_channels, 1)

        # Projection head for the contrastive loss, applied to the bottleneck
        # (deepest, most anatomy-semantic) feature map.
        self.proj_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c * 4, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = x + self.out_conv(d1)

        if return_embedding:
            embedding = F.normalize(self.proj_head(e3), dim=-1)
            return out, embedding
        return out


def ascon_contrastive_loss(embeddings: torch.Tensor, patch_ids: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be (B,D); got {tuple(embeddings.shape)}")
    b = embeddings.shape[0]
    sim = embeddings @ embeddings.t() / temperature  # (B,B)
    # A large finite sentinel, not -inf: multiplying literal -inf by a
    # False (=0) mask entry produces NaN (0 * -inf), which silently poisons
    # the whole row sum below -- found via this rewrite's test suite (a
    # perfect-positive-pair case reliably reproduced it).
    sim = sim.masked_fill(torch.eye(b, dtype=torch.bool, device=sim.device), -1e9)

    same = patch_ids.view(-1, 1) == patch_ids.view(1, -1)
    same.fill_diagonal_(False)

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_counts = same.sum(dim=1).clamp(min=1)
    loss_per_item = -(log_prob * same).sum(dim=1) / pos_counts
    valid = same.sum(dim=1) > 0
    if valid.sum() == 0:
        return embeddings.new_zeros(())
    return loss_per_item[valid].mean()
