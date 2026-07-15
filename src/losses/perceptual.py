from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16

_LAYER_MAP = {
    "relu1_2": 3,
    "relu2_2": 8,
    "relu3_3": 15,
    "relu4_3": 22,
    "relu5_3": 29,
}


class VGGPerceptualLoss(nn.Module):

    def __init__(self, layers: List[str]):
        super().__init__()
        feats = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features
        self.vgg = feats.eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        self.layers = layers
        self.max_idx = max(_LAYER_MAP[l] for l in layers)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,H,W) in [-1,1]
        x = (x + 1.0) / 2.0
        x = x.clamp(0, 1)
        x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self._prep(x)
        y = self._prep(y)

        loss = x.new_zeros(())
        fx, fy = x, y
        for i, layer in enumerate(self.vgg):
            if i > self.max_idx:
                break
            fx = layer(fx)
            fy = layer(fy)
            for name in self.layers:
                if i == _LAYER_MAP[name]:
                    loss = loss + F.l1_loss(fx, fy)
        return loss
