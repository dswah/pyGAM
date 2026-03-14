"""Tests for distribution classes, with focus on weight handling correctness."""

import numpy as np
import pytest
from scipy.stats import norm

from pygam.distributions import NormalDist


class TestNormalDistLogPdf:
    """Verify that NormalDist.log_pdf matches the GLM convention Var = σ²/w."""

    def test_weighted_logpdf_matches_scipy(self):
        """Weighted log-pdf should agree with scipy.stats.norm.logpdf
        using sd = scale / sqrt(weights)."""
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.1, 1.9, 3.2])
        weights = np.array([1.0, 2.0, 0.5])

        dist = NormalDist(scale=1.0)

        pygam_val = dist.log_pdf(y, mu, weights)
        scipy_val = norm.logpdf(y, loc=mu, scale=1.0 / np.sqrt(weights))

        np.testing.assert_allclose(pygam_val, scipy_val)

    def test_unit_weights_match_unweighted(self):
        """Unit weights should produce the same result as no weights."""
        y = np.array([0.5, 1.5, 2.5])
        mu = np.array([0.4, 1.6, 2.4])

        dist = NormalDist(scale=1.0)

        val_none = dist.log_pdf(y, mu, weights=None)
        val_ones = dist.log_pdf(y, mu, weights=np.ones(3))

        np.testing.assert_allclose(val_none, val_ones)

    def test_scalar_weight(self):
        """A scalar weight should broadcast correctly."""
        y = np.array([1.0, 2.0])
        mu = np.array([1.0, 2.0])

        dist = NormalDist(scale=1.0)

        # With weight=1 and y==mu, log_pdf should equal log(1/sqrt(2*pi))
        val = dist.log_pdf(y, mu, weights=np.array([1.0, 1.0]))
        expected = norm.logpdf(0.0, 0.0, 1.0)  # -0.9189...
        np.testing.assert_allclose(val, expected)

    def test_high_weight_concentrates_density(self):
        """Higher weight should yield higher log-pdf at the mean
        (narrower distribution)."""
        y = np.array([1.0])
        mu = np.array([1.0])

        dist = NormalDist(scale=1.0)

        lp_low = dist.log_pdf(y, mu, weights=np.array([1.0]))
        lp_high = dist.log_pdf(y, mu, weights=np.array([100.0]))

        # Higher weight → smaller sd → higher density at the mode
        assert lp_high > lp_low

    def test_nonunit_scale(self):
        """Non-unit scale should be handled correctly with weights."""
        y = np.array([2.0, 4.0])
        mu = np.array([2.1, 3.8])
        weights = np.array([1.0, 3.0])
        scale = 2.5

        dist = NormalDist(scale=scale)

        pygam_val = dist.log_pdf(y, mu, weights)
        scipy_val = norm.logpdf(y, loc=mu, scale=scale / np.sqrt(weights))

        np.testing.assert_allclose(pygam_val, scipy_val)
