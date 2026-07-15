import torch

from src.ct.operators import ParallelBeamGeometry
from src.ct.operators_torch import TorchRadonProjector
from src.losses.physics import PhysicsConsistencyLoss
from src.metrics.sinogram import sinogram_residual


def _projector():
    geom = ParallelBeamGeometry(image_size=16, angles=8, det_count=16)
    return TorchRadonProjector(geom, device=torch.device("cpu"))


def test_physics_consistency_loss_zero_for_matching_sinogram():
    proj = _projector()
    x = torch.randn(1, 1, 16, 16)
    y = proj.A(x)
    loss_fn = PhysicsConsistencyLoss()
    loss = loss_fn(proj, x, y)
    assert loss.item() < 1e-8


def test_physics_consistency_loss_nonzero_for_mismatched_sinogram():
    proj = _projector()
    x = torch.randn(1, 1, 16, 16)
    y_wrong = torch.randn(1, 8, 16)
    loss_fn = PhysicsConsistencyLoss()
    loss = loss_fn(proj, x, y_wrong)
    assert loss.item() > 0


def test_sinogram_residual_zero_for_self_consistent_input():
    proj = _projector()
    x = torch.randn(1, 1, 16, 16)
    y = proj.A(x)
    res = sinogram_residual(proj, x, y)
    assert res.l2 < 1e-3
    assert res.rmse < 1e-3


def test_sinogram_residual_positive_for_inconsistent_input():
    proj = _projector()
    x = torch.randn(1, 1, 16, 16)
    y_wrong = torch.randn(1, 8, 16) * 10
    res = sinogram_residual(proj, x, y_wrong)
    assert res.l2 > 0
    assert res.rmse > 0
