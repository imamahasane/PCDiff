
from __future__ import annotations

import torch

from src.ct.operators import ParallelBeamGeometry
from src.ct.operators_torch import TorchRadonProjector


def _shepp_logan_like(size: int) -> torch.Tensor:

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij"
    )
    img = torch.zeros(size, size)
    img[(xx / 0.9) ** 2 + (yy / 0.65) ** 2 <= 1] = 1.0
    img[((xx - 0.2) / 0.2) ** 2 + (yy / 0.2) ** 2 <= 1] = 0.6
    img[((xx + 0.2) / 0.15) ** 2 + ((yy - 0.1) / 0.15) ** 2 <= 1] = 0.3
    return img


def test_forward_adjoint_shapes():
    geom = ParallelBeamGeometry(image_size=32, angles=16, det_count=32)
    proj = TorchRadonProjector(geom, device=torch.device("cpu"))

    x = _shepp_logan_like(32)[None, None]  # (1,1,32,32)
    y = proj.A(x)
    assert y.shape == (1, 16, 32), f"unexpected forward-projection shape {tuple(y.shape)}"
    assert torch.isfinite(y).all()

    c = proj.AT(y)
    assert c.shape == (1, 1, 32, 32), f"unexpected adjoint shape {tuple(c.shape)}"
    assert torch.isfinite(c).all()
    assert c.abs().sum() > 0, "adjoint backprojection is degenerately all-zero"


def test_forward_adjoint_dot_product_consistency():

    torch.manual_seed(0)
    geom = ParallelBeamGeometry(image_size=24, angles=12, det_count=24)
    proj = TorchRadonProjector(geom, device=torch.device("cpu"))

    x = torch.randn(2, 1, 24, 24)
    y = torch.randn(2, 12, 24)

    lhs = (proj.A(x) * y).sum().item()
    rhs = (x * proj.AT(y)).sum().item()

    rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-8)
    assert rel_err < 0.35, (
        f"forward/adjoint dot-product mismatch too large (rel_err={rel_err:.3f}); "
        "this operator pair is not behaving as an approximate adjoint at all, not just "
        "imprecisely -- investigate before using it for anything, even as a CPU fallback."
    )


def test_zero_input_gives_zero_output():
    geom = ParallelBeamGeometry(image_size=16, angles=8, det_count=16)
    proj = TorchRadonProjector(geom, device=torch.device("cpu"))
    x = torch.zeros(1, 1, 16, 16)
    y = proj.A(x)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)
