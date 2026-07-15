from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .operators import ParallelBeamGeometry


def _rotation_grid(theta: torch.Tensor, size: int, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    zero = torch.zeros_like(theta)
    mat = torch.stack(
        [
            torch.stack([cos_t, -sin_t, zero], dim=-1),
            torch.stack([sin_t, cos_t, zero], dim=-1),
        ],
        dim=-2,
    ).to(dtype)
    mat = mat.expand(batch, 2, 3)
    return F.affine_grid(mat, size=(batch, 1, size, size), align_corners=False)


def _pad_to(x: torch.Tensor, size: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    if h == size and w == size:
        return x
    pad_h, pad_w = size - h, size - w
    top, left = pad_h // 2, pad_w // 2
    return F.pad(x, [left, pad_w - left, top, pad_h - top])


def _crop_from(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    size = x.shape[-1]
    if size == h and size == w:
        return x
    top, left = (size - h) // 2, (size - w) // 2
    return x[..., top : top + h, left : left + w]


class TorchRadonProjector(nn.Module):
    
    def __init__(self, geom: ParallelBeamGeometry, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.geom = geom
        self.device = device
        self.dtype = dtype
        self.angles = torch.linspace(0.0, math.pi, geom.angles, device=device, dtype=dtype)

    def A(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"x must be (B,1,H,W); got {tuple(x.shape)}")
        b, _, h, w = x.shape
        d = self.geom.det_count
        pad_size = max(d, h, w)
        x_p = _pad_to(x, pad_size)

        projections = []
        for theta in self.angles:
            grid = _rotation_grid(theta.expand(b), pad_size, b, x.device, x.dtype)
            rotated = F.grid_sample(x_p, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            proj = rotated.sum(dim=2).squeeze(1)  # (B, pad_size)
            if pad_size != d:
                proj = _crop_from(proj[:, None, None, :], 1, d).squeeze(1).squeeze(1)
            projections.append(proj)
        return torch.stack(projections, dim=1)  # (B, angles, det_count)

    def AT(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim != 3:
            raise ValueError(f"y must be (B,angles,det); got {tuple(y.shape)}")
        b, num_angles, d = y.shape
        h = w = self.geom.image_size
        pad_size = max(d, h, w)

        acc = torch.zeros(b, 1, pad_size, pad_size, device=y.device, dtype=y.dtype)
        for i in range(num_angles):
            theta = self.angles[i]
            proj = y[:, i, :]  # (B, D)
            if pad_size != d:
                proj_p = F.pad(proj, [(pad_size - d) // 2, pad_size - d - (pad_size - d) // 2])
            else:
                proj_p = proj
            smeared = proj_p[:, None, None, :].expand(b, 1, pad_size, pad_size)
            grid = _rotation_grid((-theta).expand(b), pad_size, b, y.device, y.dtype)
            rotated_back = F.grid_sample(smeared, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            acc = acc + rotated_back
        return _crop_from(acc, h, w)
