import numpy as np

from pygam.distributions import InvGaussDist


def test_inv_gauss_dist_sample_no_scale():
    # Issue #505: InvGaussDist.sample crashes when scale=None
    dist = InvGaussDist(scale=None)

    # We expect `sample` to fall back to scale=1.0 and not raise a TypeError
    samples = dist.sample(mu=1.0)

    # Assert we get a valid output
    assert samples is not None
    assert isinstance(samples, (float, np.ndarray))
