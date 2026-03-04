import numpy as np
import pytest
import scipy.stats

from pygam.distributions import NormalDist


def test_normal_log_pdf_weights_match_scipy():
    """
    check that NormalDist.log_pdf with non-unit weights agrees with scipy.

    the GLM convention is Var = scale^2 / w, so SD = scale / sqrt(w).
    """
    dist = NormalDist(scale=1.0)
    y = np.array([0.0])
    mu = np.array([0.0])
    w = np.array([4.0])

    result = dist.log_pdf(y, mu, weights=w)
    ref = scipy.stats.norm.logpdf(0.0, loc=0.0, scale=1.0 / np.sqrt(4.0))

    np.testing.assert_allclose(result, ref, rtol=1e-12)


def test_normal_log_pdf_unit_weights():
    """
    weights=1 should give the same result as weights=None
    """
    dist = NormalDist(scale=2.0)
    y = np.array([1.0, 2.0, 3.0])
    mu = np.array([1.5, 2.5, 2.0])

    a = dist.log_pdf(y, mu, weights=None)
    b = dist.log_pdf(y, mu, weights=np.ones(3))

    np.testing.assert_allclose(a, b, rtol=1e-12)


@pytest.mark.parametrize("scale", [0.5, 1.0, 3.0])
@pytest.mark.parametrize("w", [0.25, 1.0, 4.0, 16.0])
def test_normal_log_pdf_various_scales_and_weights(scale, w):
    """
    check against scipy for a range of scale and weight combos
    """
    dist = NormalDist(scale=scale)
    y = np.array([1.0])
    mu = np.array([0.0])

    result = dist.log_pdf(y, mu, weights=np.array([w]))
    sd = scale / np.sqrt(w)
    ref = scipy.stats.norm.logpdf(1.0, loc=0.0, scale=sd)

    np.testing.assert_allclose(result, ref, rtol=1e-12)
