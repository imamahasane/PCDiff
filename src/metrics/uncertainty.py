from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


def mean_and_std(samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """samples: (K, ...) independent conditional-diffusion reconstructions."""
    mean = samples.mean(dim=0)
    std = samples.std(dim=0, unbiased=True)
    return mean, std


def pearson_r(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.flatten()
    b = b.flatten()
    a = a - a.mean()
    b = b - b.mean()
    r = (a * b).mean() / (a.std(unbiased=False) * b.std(unbiased=False) + eps)
    return float(r.item())


@dataclass
class CoverageResult:
    coverage95: float
    ece: float


def coverage_and_ece(error: torch.Tensor, sigma: torch.Tensor, z: float = 1.96, eps: float = 1e-8) -> CoverageResult:
    
    err = error.abs()
    bound = z * sigma.abs().clamp(min=eps)
    covered = (err <= bound).float()
    coverage95 = float(covered.mean().item())

    sig = sigma.flatten().detach().cpu().numpy()
    errn = err.flatten().detach().cpu().numpy()
    bins = np.quantile(sig, np.linspace(0, 1, 11))
    ece = 0.0
    total = len(sig)
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        m = (sig >= lo) & (sig <= hi if i == 9 else sig < hi)
        if m.sum() == 0:
            continue
        avg_sig = sig[m].mean()
        avg_err = errn[m].mean()
        ece += (m.sum() / total) * abs(avg_err - avg_sig)
    return CoverageResult(coverage95=coverage95, ece=float(ece))
