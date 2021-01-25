from math import pi
import numpy as np

from lie_learn.representations.SO3.wigner_d import wigner_D_matrix

from se3_transformer_pytorch.utils import exists, default
from se3_transformer_pytorch.spherical_harmonics import get_spherical_harmonics, clear_spherical_harmonics_cache

def spherical_harmonics(order, alpha, beta, dtype=None):
    """
    spherical harmonics
    - compatible with irr_repr and compose

    computation time: executing 1000 times with array length 1 took 0.29 seconds;
    executing it once with array of length 1000 took 0.0022 seconds
    """
    return get_spherical_harmonics(order, theta = (pi - beta), phi = alpha) 

def irr_repr(order, alpha, beta, gamma, dtype = None):
    """
    irreducible representation of SO3
    - compatible with compose and spherical_harmonics
    """
    dtype = default(dtype, torch.get_default_type())
    alpha, beta, gamma = map(np.array, (alpha, beta, gamma))
    return torch.tensor(wigner_D_matrix(order, alpha, beta, gamma), dtype = dtype)
