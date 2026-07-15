from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AugmentConfig:
    rotate: bool = True
    hflip: bool = True
    brightness_contrast: bool = True
    brightness: float = 0.1
    contrast: float = 0.1


def random_augment(x: torch.Tensor, cfg: AugmentConfig) -> torch.Tensor:
    
    squeeze_back = x.ndim == 3
    if squeeze_back:
        x = x.unsqueeze(0)
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"Expected (B,1,H,W); got {tuple(x.shape)}")

    out = x.clone()
    b = out.shape[0]

    if cfg.hflip:
        mask = torch.rand(b, device=out.device) < 0.5
        if mask.any():
            out[mask] = torch.flip(out[mask], dims=[3])

    if cfg.rotate:
        k = torch.randint(0, 4, (b,), device=out.device)
        for i in range(b):
            out[i] = torch.rot90(out[i], int(k[i].item()), dims=[1, 2])

    if cfg.brightness_contrast:
        bright = (torch.rand(b, device=out.device) * 2 - 1) * cfg.brightness
        contrast = 1.0 + (torch.rand(b, device=out.device) * 2 - 1) * cfg.contrast
        out = out * contrast.view(b, 1, 1, 1) + bright.view(b, 1, 1, 1)

    return out.squeeze(0) if squeeze_back else out
