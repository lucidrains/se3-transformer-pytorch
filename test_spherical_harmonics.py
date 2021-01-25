import time
import torch
import numpy as np

from lie_learn.representations.SO3.spherical_harmonics import sh
from se3_transformer_pytorch.spherical_harmonics import get_spherical_harmonics_element

def benchmark(fn):
    def inner(*args, **kwargs):
        start = time.time()
        res = fn(*args, **kwargs)
        diff = time.time() - start
        return diff, res
    return inner

def test_spherical_harmonics():
    dtype = torch.float64

    theta = 0.1 * torch.randn(32, 1024, 10, dtype=dtype)
    phi = 0.1 * torch.randn(32, 1024, 10, dtype=dtype)

    s0 = s1 = 0
    max_error = -1.

    for l in range(8):
        for m in range(-l, l + 1):
            start = time.time()

            diff, y = benchmark(get_spherical_harmonics_element)(l, m, theta, phi)
            y = y.type(torch.float32)
            s0 += diff

            diff, z = benchmark(sh)(l, m, theta, phi)
            s1 += diff

            error = np.mean(np.abs((y.cpu().numpy() - z) / z))
            max_error = max(max_error, error)
            print(f"l: {l}, m: {m} ", error)

    time_diff_ratio = s0 / s1

    assert max_error < 1e-4, 'maximum error must be less than 1e-3'
    assert time_diff_ratio < 1., 'spherical harmonics must be faster than the one offered by lie_learn'

    print(f"Max error: {max_error}")
    print(f"Time diff: {time_diff_ratio}")
