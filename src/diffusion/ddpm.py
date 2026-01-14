from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .schedule import DiffusionSchedule

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    
    out = a.gather(0, t)
    return out.view(-1, *([1] * (len(x_shape) - 1)))

@dataclass
class DDPMOutput:
    eps_pred: torch.Tensor
    x0_pred: torch.Tensor

class PhysicsConditionedDDPM(nn.Module):


    def __init__(self, denoiser: nn.Module, schedule: DiffusionSchedule):
        super().__init__()
        self.denoiser = denoiser
        self.schedule = schedule

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        if eps is None:
            eps = torch.randn_like(x0)
        a_bar = extract(self.schedule.alpha_bars, t, x0.shape)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * eps

    def predict_eps_and_x0(self, x_t: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> DDPMOutput:
        inp = torch.cat([x_t, c], dim=1)
        eps_pred = self.denoiser(inp, t)
        a_bar = extract(self.schedule.alpha_bars, t, x_t.shape)
        x0_pred = (x_t - torch.sqrt(1.0 - a_bar) * eps_pred) / torch.sqrt(a_bar)
        return DDPMOutput(eps_pred=eps_pred, x0_pred=x0_pred)

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        
        out = self.predict_eps_and_x0(x_t, c, t)
        beta_t = extract(self.schedule.betas, t, x_t.shape)
        alpha_t = extract(self.schedule.alphas, t, x_t.shape)
        a_bar_t = extract(self.schedule.alpha_bars, t, x_t.shape)

        # Eq. line 19 in Algorithm 1: mu_t = 1/sqrt(alpha_t) (x_t - (1-alpha_t)/sqrt(1-a_bar_t) * eps_hat)
        mu = (x_t - (1.0 - alpha_t) / torch.sqrt(1.0 - a_bar_t) * out.eps_pred) / torch.sqrt(alpha_t)

        # sigma_t^2 from DDPM posterior variance
        var = extract(self.schedule.posterior_variance, t, x_t.shape)
        if (t == 0).all():
            return mu  # not used
        noise = torch.randn_like(x_t)
        return mu + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, c: torch.Tensor, shape: Tuple[int, int, int, int]) -> torch.Tensor:
        
        device = c.device
        x = torch.randn(shape, device=device)
        T = self.schedule.betas.shape[0]
        for ti in reversed(range(T)):
            t = torch.full((shape[0],), ti, device=device, dtype=torch.long)
            x = self.p_sample(x, c, t)
        return x
