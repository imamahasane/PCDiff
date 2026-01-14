from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim

def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 2.0) -> float:
    mse = F.mse_loss(x, y).item()
    if mse == 0:
        return float("inf")
    return 20 * np.log10(data_range) - 10 * np.log10(mse)

def ssim_torch(x: torch.Tensor, y: torch.Tensor, data_range: float = 2.0) -> float:
    
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    # assume (1,H,W)
    return float(ssim(x_np[0], y_np[0], data_range=data_range))
