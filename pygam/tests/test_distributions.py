"""Tests for pygam.distributions."""

import numpy as np
import scipy.stats

from pygam.distributions import NormalDist


class TestNormalDistLogPdf:
    """Test that NormalDist.log_pdf handles weights correctly.

    The GLM convention used throughout pyGAM is Var[Y] = scale^2 / w,
    so SD = scale / sqrt(w). log_pdf should use the same convention.
    """

    def test_weights_equal_one_unchanged(self):
        dist = NormalDist(scale=1.0)
        y = np.array([1.0])
        mu = np.array([0.0])
        result = dist.log_pdf(y, mu, weights=np.array([1.0]))
        expected = scipy.stats.norm.logpdf(1.0, loc=0.0, scale=1.0)
        np.testing.assert_allclose(result, expected)

    def test_weights_greater_than_one(self):
        dist = NormalDist(scale=1.0)
        y = np.array([0.0])
        mu = np.array([0.0])
        w = np.array([4.0])
        result = dist.log_pdf(y, mu, weights=w)
        # SD = scale / sqrt(w) = 1 / 2 = 0.5
        expected = scipy.stats.norm.logpdf(0.0, loc=0.0, scale=0.5)
        np.testing.assert_allclose(result, expected)

    def test_weights_vectorized(self):
        dist = NormalDist(scale=2.0)
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([0.0, 0.0, 0.0])
        w = np.array([1.0, 4.0, 9.0])
        result = dist.log_pdf(y, mu, weights=w)
        expected = scipy.stats.norm.logpdf(
            y, loc=mu, scale=2.0 / np.sqrt(w)
        )
        np.testing.assert_allclose(result, expected)

    def test_no_weights_defaults_to_unweighted(self):
        dist = NormalDist(scale=1.0)
        y = np.array([0.5])
        mu = np.array([0.0])
        with_weights = dist.log_pdf(y, mu, weights=np.array([1.0]))
        without_weights = dist.log_pdf(y, mu, weights=None)
        np.testing.assert_allclose(with_weights, without_weights)
