from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim


def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float) -> float:
    
    mse = F.mse_loss(x, y).item()
    if mse == 0:
        return float("inf")
    return 20 * np.log10(data_range) - 10 * np.log10(mse)


def ssim_torch(x: torch.Tensor, y: torch.Tensor, data_range: float) -> float:
    """x, y: (1,H,W) single-channel images."""
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(f"x must be (1,H,W); got {tuple(x.shape)}")
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    return float(ssim(x_np[0], y_np[0], data_range=data_range))
