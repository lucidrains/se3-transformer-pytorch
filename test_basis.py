import torch
from se3_transformer_pytorch.basis import get_basis

def test_basis():
    max_degree = 3
    x = torch.randn(1024, 3)
    basis = get_basis(x, max_degree)
    assert len(basis.keys()) == (max_degree + 1) ** 2, 'correct number of basis kernels'
