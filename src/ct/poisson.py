from __future__ import annotations

import torch


def add_poisson_noise_to_sinogram(clean_sino: torch.Tensor, i0: float = 1e4, eps: float = 1e-6) -> torch.Tensor:
    
    if clean_sino.ndim != 3:
        raise ValueError(f"clean_sino must be (B,angles,det); got {tuple(clean_sino.shape)}")
    intensity = i0 * torch.exp(-clean_sino)
    intensity_noisy = torch.poisson(intensity)
    return -torch.log(torch.clamp(intensity_noisy / i0, min=eps, max=1.0))
