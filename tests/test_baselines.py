import torch

from src.models.baselines.ascon import ASCONDenoiser, ascon_contrastive_loss
from src.models.baselines.corediff import CoreDiffModel, make_generalized_schedule
from src.models.baselines.dugan import DUGANDiscriminators, DUGANGenerator, image_gradient
from src.models.baselines.hformer import Hformer
from src.models.baselines.redcnn import REDCNN


def test_redcnn_forward_shape_and_size_preserving():
    model = REDCNN()
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_dugan_generator_and_discriminators():
    gen = DUGANGenerator(base_channels=8)
    x = torch.randn(2, 1, 64, 64)
    fake = gen(x)
    assert fake.shape == x.shape

    disc = DUGANDiscriminators(base_channels=8)
    img_score, grad_score = disc(fake)
    assert img_score.ndim == 4
    assert grad_score.ndim == 4


def test_image_gradient_zero_for_constant_image_interior():
    # Border pixels see a real (not buggy) discontinuity from zero-padding;
    # check the interior only, away from that expected edge effect.
    x = torch.ones(1, 1, 16, 16) * 0.5
    g = image_gradient(x)
    interior = g[:, :, 2:-2, 2:-2]
    assert torch.allclose(interior, torch.zeros_like(interior), atol=1e-5)


def test_hformer_forward_shape():
    model = Hformer(base_channels=8, num_transformer_blocks=1, num_heads=2)
    x = torch.randn(2, 1, 32, 32)
    out = model(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_ascon_forward_and_contrastive_loss():
    model = ASCONDenoiser(base_channels=8, embed_dim=16)
    x = torch.randn(4, 1, 32, 32)
    out, emb = model(x, return_embedding=True)
    assert out.shape == x.shape
    assert emb.shape == (4, 16)

    patch_ids = torch.arange(4).repeat(2)
    embeddings = torch.cat([emb, emb], dim=0)  # perfect positive pairs (identical embeddings)
    loss = ascon_contrastive_loss(embeddings, patch_ids)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_corediff_predict_and_sample_shapes():
    model = CoreDiffModel(image_size=32, base_channels=8)
    schedule = make_generalized_schedule(T=5)
    x_ldct = torch.randn(1, 1, 32, 32)
    x_t = torch.randn(1, 1, 32, 32)
    t = torch.zeros(1, dtype=torch.long)

    x0_pred = model.predict_x0(x_t, x_ldct, t)
    assert x0_pred.shape == x_ldct.shape

    sampled = model.sample(x_ldct, schedule)
    assert sampled.shape == x_ldct.shape
    assert torch.isfinite(sampled).all()
