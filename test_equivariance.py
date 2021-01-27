import torch
from se3_transformer_pytorch.se3_transformer_pytorch import SE3Transformer

def test_transformer():
    model = SE3Transformer(
        dim = 64,
        num_degrees = 2
    )

    feats = torch.randn(1, 32, 64)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    out = model(feats, coors, mask)
    assert out['0'].shape == (1, 32, 64, 1), 'output must be of the right shape'

def test_equivariance():
    # todo
    assert True