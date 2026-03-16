"""Tests for pygam.distributions."""

import numpy as np
import scipy.stats

from pygam.distributions import InvGaussDist, NormalDist


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
        expected = scipy.stats.norm.logpdf(y, loc=mu, scale=2.0 / np.sqrt(w))
        np.testing.assert_allclose(result, expected)

    def test_no_weights_defaults_to_unweighted(self):
        dist = NormalDist(scale=1.0)
        y = np.array([0.5])
        mu = np.array([0.0])
        with_weights = dist.log_pdf(y, mu, weights=np.array([1.0]))
        without_weights = dist.log_pdf(y, mu, weights=None)
        np.testing.assert_allclose(with_weights, without_weights)


class TestInvGaussDistLogPdf:
    """Tests for InvGaussDist.log_pdf parameterization."""

    @staticmethod
    def _closed_form_logpdf(y, mu, lam):
        return 0.5 * (np.log(lam) - np.log(2 * np.pi) - 3 * np.log(y)) - (
            lam * (y - mu) ** 2
        ) / (2 * mu**2 * y)

    def test_control_case_matches_closed_form(self):
        dist = InvGaussDist(scale=1.0)
        y = np.array([2.0])
        mu = np.array([2.0])
        w = np.array([1.0])

        result = dist.log_pdf(y, mu, weights=w)
        expected = self._closed_form_logpdf(y, mu, lam=w / 1.0)

        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_weighted_non_unit_scale_matches_closed_form(self):
        dist = InvGaussDist(scale=2.5)
        y = np.array([0.5, 1.0, 2.0, 4.0])
        mu = np.array([0.6, 1.4, 1.8, 3.2])
        w = np.array([0.5, 1.0, 2.0, 5.0])

        result = dist.log_pdf(y, mu, weights=w)
        expected = self._closed_form_logpdf(y, mu, lam=w / 2.5)

        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
