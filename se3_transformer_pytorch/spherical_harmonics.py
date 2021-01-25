from math import pi, sqrt
from functools import reduce, wraps
from operator import mul

import torch

# constants

CACHE = {}

def clear_spherical_harmonics_cache():
    CACHE.clear()

def lpmv_cache_key_fn(l, m, x):
    return (l, m)

# caching functions

def cache(cache, key_fn):
    def cache_inner(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            key_name = key_fn(*args, **kwargs)
            if key_name in cache:
                return cache[key_name]
            res = fn(*args, **kwargs)
            cache[key_name] = res
            return res

        return inner
    return cache_inner

# spherical harmonics

def semifactorial(x):
    return reduce(mul, range(x, 1, -2), 1.)

def pochhammer(x, k):
    return reduce(mul, range(x + 1, x + k), float(x))

def negative_lpmv(l, m, y):
    if m < 0:
        y *= ((-1) ** m / pochhammer(l + m + 1, -2 * m))
    return y

@cache(cache = CACHE, key_fn = lpmv_cache_key_fn)
def lpmv(l, m, x):
    """Associated Legendre function including Condon-Shortley phase.

    Args:
        m: int order 
        l: int degree
        x: float argument tensor
    Returns:
        tensor of x-shape
    """
    # Check memoized versions
    m_abs = abs(m)

    if m_abs > l:
        return None

    if l == 0:
        return torch.ones_like(x)
    
    # Check if on boundary else recurse solution down to boundary
    if m_abs == l:
        # Compute P_m^m
        y = (-1)**m_abs * semifactorial(2*m_abs-1)
        y *= torch.pow(1-x*x, m_abs/2)
        return negative_lpmv(l, m, y)
    else:
        # Recursively precompute lower degree harmonics
        lpmv(l-1, m, x)

    # Compute P_{l}^m from recursion in P_{l-1}^m and P_{l-2}^m
    # Inplace speedup
    y = ((2*l-1) / (l-m_abs)) * x * lpmv(l-1, m_abs, x)

    if l - m_abs > 1:
        y -= ((l+m_abs-1)/(l-m_abs)) * CACHE[(l-2, m_abs)]
    
    if m < 0:
        y = self.negative_lpmv(l, m, y)
    return y

def get_spherical_harmonics_element(l, m, theta, phi):
    """Tesseral spherical harmonic with Condon-Shortley phase.

    The Tesseral spherical harmonics are also known as the real spherical
    harmonics.

    Args:
        l: int for degree
        m: int for order, where -l <= m < l
        theta: collatitude or polar angle
        phi: longitude or azimuth
    Returns:
        tensor of shape theta
    """
    assert abs(m) <= l, "absolute value of order m must be <= degree l"

    N = sqrt((2*l+1) / (4*pi))
    leg = lpmv(l, abs(m), torch.cos(theta))
    if m == 0:
        return N*leg

    if m > 0:
        Y = torch.cos(m*phi) * leg
    else:
        Y = torch.sin(abs(m)*phi) * leg

    N *= sqrt(2. / pochhammer(l-abs(m)+1, 2*abs(m)))
    Y *= N
    return Y

def get_spherical_harmonics(l, theta, phi):
    """Tesseral harmonic with Condon-Shortley phase.

    The Tesseral spherical harmonics are also known as the real spherical
    harmonics.

    Args:
        l: int for degree
        theta: collatitude or polar angle
        phi: longitude or azimuth
    Returns:
        tensor of shape [*theta.shape, 2*l+1]
    """
    results = []
    for m in range(-l, l+1):
        el = get_element(l, m, theta, phi)
        results.append(el)
    return torch.stack(results, dim = -1)
