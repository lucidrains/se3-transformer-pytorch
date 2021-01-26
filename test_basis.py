import torch
from se3_transformer_pytorch.basis import get_basis, get_R_tensor, basis_transformation_Q_J
from se3_transformer_pytorch.irr_repr import irr_repr

def test_basis():
    max_degree = 3
    x = torch.randn(2, 1024, 3)
    basis = get_basis(x, max_degree)
    assert len(basis.keys()) == (max_degree + 1) ** 2, 'correct number of basis kernels'

def test_basis_transformation_Q_J():
    rand_angles = torch.rand(4, 3)
    J, order_out, order_in = 1, 1, 1
    Q_J = basis_transformation_Q_J(J, order_in, order_out).float()
    assert all(torch.allclose(get_R_tensor(order_out, order_in, a, b, c) @ Q_J, Q_J @ irr_repr(J, a, b, c)) for a, b, c in rand_angles)
