from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

try:
    from torch_radon import Radon
except Exception:  # pragma: no cover - torch-radon is CUDA-only and often fails to import
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
                "torch-radon is required for CTProjector (production CUDA path). "
                "Install it via `pip install torch-radon`, or use "
                "src.ct.operators_torch.TorchRadonProjector as a CPU-compatible "
                "fallback (slower, no CUDA dependency)."
            )
        self.geom = geom
        self.radon = Radon(
            geom.image_size,
            torch.linspace(0.0, torch.pi, geom.angles, device=device),
            det_count=geom.det_count,
        )

    def extra_repr(self) -> str:
        return f"ParallelBeamGeometry(image_size={self.geom.image_size}, angles={self.geom.angles}, det_count={self.geom.det_count})"

    def A(self, x: torch.Tensor) -> torch.Tensor:
        """Forward projection y = A(x). x: (B,1,H,W) -> y: (B,angles,det)."""
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"x must be (B,1,H,W); got {tuple(x.shape)}")
        return self.radon.forward(x[:, 0])

    def AT(self, y: torch.Tensor) -> torch.Tensor:
        """Adjoint backprojection c = A*(y). y: (B,angles,det) -> c: (B,1,H,W)."""
        if y.ndim != 3:
            raise ValueError(f"y must be (B,angles,det); got {tuple(y.shape)}")
        x_bp = self.radon.backward(y)
        return x_bp[:, None]
