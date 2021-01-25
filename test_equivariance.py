import torch
from se3_transformer_pytorch.se3_transformer_pytorch import SE3Transformer

def test_transformer():
    model = SE3Transformer(
        dim = 512,
        num_degrees = 4
    )

    feats = torch.randn(1, 1024, 512)
    coors = torch.randn(1, 1024, 3)
    mask  = torch.ones(1, 1024).bool()

    out = model(feats, coors, mask)
    assert out.shape == (1, 1024, 512), 'output must be of the right shape'

def test_equivariance():
    # todo
    assert True