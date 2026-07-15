from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    posterior_variance: torch.Tensor


def make_beta_schedule(
    schedule: Literal["linear"] = "linear",
    T: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: torch.device | None = None,
) -> torch.Tensor:
    if schedule != "linear":
        raise ValueError(f"Unsupported beta schedule: {schedule}")
    return torch.linspace(beta_start, beta_end, T, device=device)


def make_ddpm_schedule(
    T: int,
    beta_schedule: str,
    beta_start: float,
    beta_end: float,
    device: torch.device,
) -> DiffusionSchedule:
    betas = make_beta_schedule(beta_schedule, T, beta_start, beta_end, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    alpha_bars_prev = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]], dim=0)
    posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        posterior_variance=posterior_variance,
    )
