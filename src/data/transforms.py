from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

@dataclass
class AugmentConfig:
    rotate: bool = True
    hflip: bool = True
    brightness_contrast: bool = True
    brightness: float = 0.1
    contrast: float = 0.1

def random_augment(x: torch.Tensor, cfg: AugmentConfig) -> torch.Tensor:

    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"Expected (B,1,H,W); got {tuple(x.shape)}")

    B = x.shape[0]
    out = x

    if cfg.hflip:
        mask = torch.rand(B, device=x.device) < 0.5
        if mask.any():
            out[mask] = torch.flip(out[mask], dims=[3])

    if cfg.rotate:
        # multiples of 90 degrees to avoid interpolation artifacts
        k = torch.randint(0, 4, (B,), device=x.device)
        for i in range(B):
            out[i] = torch.rot90(out[i], int(k[i].item()), dims=[1, 2])

    if cfg.brightness_contrast:
        # brightness and contrast jitter in a simple, deterministic-safe way
        b = (torch.rand(B, device=x.device) * 2 - 1) * cfg.brightness
        c = 1.0 + (torch.rand(B, device=x.device) * 2 - 1) * cfg.contrast
        out = out * c.view(B,1,1,1) + b.view(B,1,1,1)

    return out.squeeze(0) if x.shape[0] == 1 else out
