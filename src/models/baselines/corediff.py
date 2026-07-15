from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.unet import UNetConfig, UNetModel


class ContextualErrorModulation(nn.Module):

    def __init__(self, channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

    def forward(self, x_pred: torch.Tensor, x_ldct: torch.Tensor) -> torch.Tensor:
        correction = self.net(torch.cat([x_pred, x_ldct], dim=1))
        return x_pred + correction


@dataclass
class GeneralizedDiffusionSchedule:
    
    gammas: torch.Tensor  # interpolation weight toward x_ldct
    sigmas: torch.Tensor  # residual stochastic noise scale


def make_generalized_schedule(T: int, sigma_max: float = 0.05, device: torch.device | None = None) -> GeneralizedDiffusionSchedule:
    gammas = torch.linspace(1.0, 0.0, T, device=device)
    sigmas = sigma_max * torch.sin(torch.linspace(0.0, torch.pi, T, device=device))
    return GeneralizedDiffusionSchedule(gammas=gammas, sigmas=sigmas)


class CoreDiffModel(nn.Module):

    def __init__(self, image_size: int, base_channels: int = 64):
        super().__init__()
        unet_cfg = UNetConfig(
            in_channels=2,  # concat(x_t, x_ldct context)
            out_channels=1,
            base_channels=base_channels,
            channel_mult=(1, 2, 2, 4),
            num_res_blocks=2,
            attention_resolutions=(16,),
            num_heads=4,
            dropout=0.0,
        )
        self.denoiser = UNetModel(unet_cfg, image_size=image_size)
        self.cem = ContextualErrorModulation()

    def predict_x0(self, x_t: torch.Tensor, x_ldct: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x_t, x_ldct], dim=1)
        x0_pred = self.denoiser(inp, t)
        return self.cem(x0_pred, x_ldct)

    @torch.no_grad()
    def sample(self, x_ldct: torch.Tensor, schedule: GeneralizedDiffusionSchedule) -> torch.Tensor:
        
        T = schedule.gammas.shape[0]
        x_t = x_ldct + schedule.sigmas[0] * torch.randn_like(x_ldct)
        for ti in reversed(range(T)):
            t = torch.full((x_ldct.shape[0],), ti, device=x_ldct.device, dtype=torch.long)
            x0_pred = self.predict_x0(x_t, x_ldct, t)
            gamma = schedule.gammas[ti]
            sigma = schedule.sigmas[ti]
            noise = torch.randn_like(x_t) if ti > 0 else 0.0
            x_t = (1 - gamma) * x0_pred + gamma * x_ldct + sigma * noise
        return x_t
