from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SinogramResidual:
    l2: float
    rmse: float


def sinogram_residual(projector, x_hat0: torch.Tensor, y: torch.Tensor) -> SinogramResidual:
    
    sino_hat = projector.A(x_hat0)
    diff = sino_hat - y
    l2 = torch.linalg.vector_norm(diff).item()
    rmse = torch.sqrt(F.mse_loss(sino_hat, y)).item()
    return SinogramResidual(l2=l2, rmse=rmse)
