import torch
from torch import sin, cos
from math import pi
from functools import wraps
import numpy as np

from lie_learn.representations.SO3.wigner_d import wigner_D_matrix

from se3_transformer_pytorch.utils import exists, default, cast_torch_tensor
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
    dtype = default(dtype, torch.get_default_dtype())
    alpha, beta, gamma = map(np.array, (alpha, beta, gamma))
    return torch.tensor(wigner_D_matrix(order, alpha, beta, gamma), dtype = dtype)

@cast_torch_tensor
def rot_z(gamma):
    '''
    Rotation around Z axis
    '''
    return torch.tensor([
        [cos(gamma), -sin(gamma), 0],
        [sin(gamma), cos(gamma), 0],
        [0, 0, 1]
    ], dtype=gamma.dtype)

@cast_torch_tensor
def rot_y(beta):
    '''
    Rotation around Y axis
    '''
    return torch.tensor([
        [cos(beta), 0, sin(beta)],
        [0, 1, 0],
        [-sin(beta), 0, cos(beta)]
    ], dtype=beta.dtype)

@cast_torch_tensor
def x_to_alpha_beta(x):
    '''
    Convert point (x, y, z) on the sphere into (alpha, beta)
    '''
    x = x / torch.norm(x)
    beta = torch.acos(x[2])
    alpha = torch.atan2(x[1], x[0])
    return (alpha, beta)

def rot(alpha, beta, gamma):
    '''
    ZYZ Euler angles rotation
    '''
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)

def compose(a1, b1, c1, a2, b2, c2):
    """
    (a, b, c) = (a1, b1, c1) composed with (a2, b2, c2)
    """
    comp = rot(a1, b1, c1) @ rot(a2, b2, c2)
    xyz = comp @ torch.tensor([0, 0, 1.])
    a, b = x_to_alpha_beta(xyz)
    rotz = rot(0, -b, -a) @ comp
    c = torch.atan2(rotz[1, 0], rotz[0, 0])
    return a, b, c
