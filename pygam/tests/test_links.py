"""Tests for ProbitLink class."""

import numpy as np
import scipy as sp

from pygam.distributions import BinomialDist
from pygam.links import LINKS, ProbitLink


class TestProbitLink:
    """Tests for ProbitLink class."""

    def test_init(self):
        """Check that ProbitLink can be instantiated."""
        link = ProbitLink()
        assert link._name == "probit"

    def test_in_links_dict(self):
        """Check that ProbitLink is in the LINKS dictionary."""
        assert "probit" in LINKS
        assert LINKS["probit"] is ProbitLink

    def test_link_returns_correct_shape(self):
        """Check that link function returns correct shape."""
        link = ProbitLink()
        dist = BinomialDist(levels=1)
        mu = np.array([0.2, 0.5, 0.8])
        lp = link.link(mu, dist)
        assert lp.shape == mu.shape

    def test_mu_returns_correct_shape(self):
        """Check that mu function returns correct shape."""
        link = ProbitLink()
        dist = BinomialDist(levels=1)
        lp = np.array([-1.0, 0.0, 1.0])
        mu = link.mu(lp, dist)
        assert mu.shape == lp.shape

    def test_gradient_returns_correct_shape(self):
        """Check that gradient function returns correct shape."""
        link = ProbitLink()
        dist = BinomialDist(levels=1)
        mu = np.array([0.2, 0.5, 0.8])
        grad = link.gradient(mu, dist)
        assert grad.shape == mu.shape

    def test_link_values_against_scipy(self):
        """Check link function values against scipy reference."""
        link = ProbitLink()
        dist = BinomialDist(levels=1)
        mu = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        lp = link.link(mu, dist)
        expected = sp.stats.norm.ppf(mu)
        np.testing.assert_array_almost_equal(lp, expected)

    def test_mu_values_against_scipy(self):
        """Check mu function values against scipy reference."""
        link = ProbitLink()
        dist = BinomialDist(levels=1)
        lp = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        mu = link.mu(lp, dist)
        expected = sp.stats.norm.cdf(lp)
        np.testing.assert_array_almost_equal(mu, expected)

    def test_link_mu_inverse(self):
        """Check that link and mu are inverses of each other."""
        link = ProbitLink()
        dist = BinomialDist(levels=1)
        mu_original = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        lp = link.link(mu_original, dist)
        mu_recovered = link.mu(lp, dist)
        np.testing.assert_array_almost_equal(mu_original, mu_recovered)

    def test_mu_link_inverse(self):
        """Check that mu and link are inverses of each other."""
        link = ProbitLink()
        dist = BinomialDist(levels=1)
        lp_original = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        mu = link.mu(lp_original, dist)
        lp_recovered = link.link(mu, dist)
        np.testing.assert_array_almost_equal(lp_original, lp_recovered)

    def test_gradient_positive_for_valid_mu(self):
        """Check that gradient is positive for valid mu values."""
        link = ProbitLink()
        dist = BinomialDist(levels=1)
        mu = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        grad = link.gradient(mu, dist)
        assert np.all(grad > 0)

    def test_gradient_at_mu_half(self):
        """Check gradient at mu=0.5 (where probit link value is 0)."""
        link = ProbitLink()
        dist = BinomialDist(levels=1)
        mu = np.array([0.5])
        grad = link.gradient(mu, dist)
        # At mu=0.5, ppf(0.5)=0, so pdf(0) = 1/sqrt(2*pi)
        expected = 1.0 / sp.stats.norm.pdf(0)
        np.testing.assert_almost_equal(grad[0], expected)

    def test_link_with_levels(self):
        """Check that link works correctly with levels > 1."""
        link = ProbitLink()
        levels = 10
        dist = BinomialDist(levels=levels)
        mu = np.array([1.0, 5.0, 9.0])  # mu in [0, levels]
        lp = link.link(mu, dist)
        expected = sp.stats.norm.ppf(mu / levels)
        np.testing.assert_array_almost_equal(lp, expected)

    def test_mu_with_levels(self):
        """Check that mu works correctly with levels > 1."""
        link = ProbitLink()
        levels = 10
        dist = BinomialDist(levels=levels)
        lp = np.array([-1.0, 0.0, 1.0])
        mu = link.mu(lp, dist)
        expected = levels * sp.stats.norm.cdf(lp)
        np.testing.assert_array_almost_equal(mu, expected)

    def test_mu_bounded(self):
        """Check that mu returns values in [0, levels]."""
        link = ProbitLink()
        dist = BinomialDist(levels=1)
        lp = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        mu = link.mu(lp, dist)
        assert np.all(mu >= 0)
        assert np.all(mu <= dist.levels)
