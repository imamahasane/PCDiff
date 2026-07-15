from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats as scipy_stats


@dataclass
class WilcoxonResult:
    statistic: float
    p_value: float
    n: int


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> WilcoxonResult:
   
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Paired arrays must match in shape: {a.shape} vs {b.shape}")
    if a.shape[0] < 1:
        raise ValueError("Need at least one paired observation")
    res = scipy_stats.wilcoxon(a, b)
    return WilcoxonResult(statistic=float(res.statistic), p_value=float(res.pvalue), n=int(a.shape[0]))


@dataclass
class BootstrapCI:
    mean: float
    lo: float
    hi: float
    ci: float
    n_boot: int


def bootstrap_ci(scores: Sequence[float], n_boot: int = 10000, ci: float = 0.95, seed: int = 0) -> BootstrapCI:
    
    x = np.asarray(scores, dtype=np.float64)
    if x.shape[0] < 2:
        raise ValueError("Need at least two observations to bootstrap")
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = x[idx].mean()
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return BootstrapCI(mean=float(x.mean()), lo=float(lo), hi=float(hi), ci=ci, n_boot=n_boot)


def bootstrap_ci_diff(a: Sequence[float], b: Sequence[float], n_boot: int = 10000, ci: float = 0.95, seed: int = 0) -> BootstrapCI:
    
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Paired arrays must match in shape: {a.shape} vs {b.shape}")
    rng = np.random.default_rng(seed)
    n = a.shape[0]
    diffs = a - b
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return BootstrapCI(mean=float(diffs.mean()), lo=float(lo), hi=float(hi), ci=ci, n_boot=n_boot)
