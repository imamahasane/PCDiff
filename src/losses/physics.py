from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsConsistencyLoss(nn.Module):
    
    def forward(self, projector, x_hat0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sino_hat = projector.A(x_hat0)
        return F.mse_loss(sino_hat, y)
