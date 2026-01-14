from __future__ import annotations

import torch

def add_poisson_noise_to_sinogram(clean_sino: torch.Tensor, i0: float = 1e4, eps: float = 1e-6) -> torch.Tensor:

    if clean_sino.ndim != 3:
        raise ValueError(f"clean_sino must be (B,angles,det); got {tuple(clean_sino.shape)}")
    I = i0 * torch.exp(-clean_sino)
    I_tilde = torch.poisson(I)
    p_tilde = -torch.log(torch.clamp(I_tilde / i0, min=eps, max=1.0))
    return p_tilde
