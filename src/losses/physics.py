from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, x_hat0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # A: object with method A(x)->sino
        sino_hat = A.A(x_hat0)
        return F.mse_loss(sino_hat, y)
