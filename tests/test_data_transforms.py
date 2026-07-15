import torch

from src.data.transforms import AugmentConfig, random_augment


def test_random_augment_does_not_mutate_input():
    
    torch.manual_seed(0)
    x = torch.randn(4, 1, 16, 16)
    x_before = x.clone()
    cfg = AugmentConfig()

    _ = random_augment(x, cfg)

    assert torch.equal(x, x_before), "random_augment mutated its input tensor in place"


def test_random_augment_output_shape_preserved():
    x = torch.randn(3, 1, 12, 12)
    out = random_augment(x, AugmentConfig())
    assert out.shape == x.shape


def test_random_augment_handles_unbatched_input():
    x = torch.randn(1, 10, 10)
    out = random_augment(x, AugmentConfig())
    assert out.shape == x.shape


def test_random_augment_rejects_wrong_shape():
    x = torch.randn(4, 3, 16, 16)  # 3 "channels" -- not the expected single-channel input
    try:
        random_augment(x, AugmentConfig())
        assert False, "expected ValueError for non-single-channel input"
    except ValueError:
        pass
