import torch
from se3_transformer_pytorch.spherical_harmonics import clear_spherical_harmonics_cache
from se3_transformer_pytorch.irr_repr import spherical_harmonics, irr_repr, compose
from se3_transformer_pytorch.utils import torch_default_dtype

@torch_default_dtype(torch.float64)
def test_irr_repr():
    """
    This test tests that
    - irr_repr
    - compose
    - spherical_harmonics
    are compatible

    Y(Z(alpha) Y(beta) Z(gamma) x) = D(alpha, beta, gamma) Y(x)
    with x = Z(a) Y(b) eta
    """
    for order in range(7):
        a, b = torch.rand(2)
        alpha, beta, gamma = torch.rand(3)

        ra, rb, _ = compose(alpha, beta, gamma, a, b, 0)
        Yrx = spherical_harmonics(order, ra, rb)
        clear_spherical_harmonics_cache()

        Y = spherical_harmonics(order, a, b)
        clear_spherical_harmonics_cache()

        DrY = irr_repr(order, alpha, beta, gamma) @ Y

        d, r = (Yrx - DrY).abs().max(), Y.abs().max()
        print(d.item(), r.item())
        assert d < 1e-10 * r, d / r
