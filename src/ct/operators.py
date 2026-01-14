from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from torch_radon import Radon
except Exception as e:  # pragma: no cover
    Radon = None

@dataclass
class ParallelBeamGeometry:
    image_size: int
    angles: int
    det_count: int

class CTProjector(nn.Module):
    def __init__(self, geom: ParallelBeamGeometry, device: torch.device):
        super().__init__()
        if Radon is None:
            raise ImportError(
                "torch-radon is required for differentiable A and A*. "
                "Install it via `pip install torch-radon`."
            )
        self.geom = geom
        self.radon = Radon(
            geom.image_size,
            torch.linspace(0.0, torch.pi, geom.angles, device=device),
            det_count=geom.det_count,
        )

    @torch.no_grad()
    def extra_repr(self) -> str:
        return f"ParallelBeamGeometry(image_size={self.geom.image_size}, angles={self.geom.angles}, det_count={self.geom.det_count})"

    def A(self, x: torch.Tensor) -> torch.Tensor:
        """Forward projection y = A(x)."""
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"x must be (B,1,H,W); got {tuple(x.shape)}")
        # torch-radon expects (B, H, W)
        y = self.radon.forward(x[:, 0])
        return y

    def AT(self, y: torch.Tensor) -> torch.Tensor:
        """Adjoint backprojection c = A*(y). (Unfiltered backprojection)"""
        if y.ndim != 3:
            raise ValueError(f"y must be (B,angles,det); got {tuple(y.shape)}")
        x_bp = self.radon.backward(y)
        return x_bp[:, None]  # (B,1,H,W)
