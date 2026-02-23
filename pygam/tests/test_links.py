"""Tests for link function classes."""

import numpy as np
import scipy as sp

from pygam.distributions import BinomialDist
from pygam.links import (
    IdentityLink,
    InverseLink,
    InvSquaredLink,
    LINKS,
    LogitLink,
    LogLink,
    ProbitLink,
)


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


class TestLogitLink:
    """Tests for LogitLink class to ensure consistency with ProbitLink tests."""

    def test_init(self):
        """Check that LogitLink can be instantiated."""
        link = LogitLink()
        assert link._name == "logit"

    def test_link_mu_inverse(self):
        """Check that link and mu are inverses of each other."""
        link = LogitLink()
        dist = BinomialDist(levels=1)
        mu_original = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        lp = link.link(mu_original, dist)
        mu_recovered = link.mu(lp, dist)
        np.testing.assert_array_almost_equal(mu_original, mu_recovered)


class TestIdentityLink:
    """Tests for IdentityLink class."""

    def test_init(self):
        """Check that IdentityLink can be instantiated."""
        link = IdentityLink()
        assert link._name == "identity"

    def test_link_is_identity(self):
        """Check that link function is identity."""
        link = IdentityLink()
        dist = BinomialDist(levels=1)
        mu = np.array([1.0, 2.0, 3.0])
        lp = link.link(mu, dist)
        np.testing.assert_array_equal(lp, mu)

    def test_mu_is_identity(self):
        """Check that mu function is identity."""
        link = IdentityLink()
        dist = BinomialDist(levels=1)
        lp = np.array([1.0, 2.0, 3.0])
        mu = link.mu(lp, dist)
        np.testing.assert_array_equal(mu, lp)

    def test_gradient_is_ones(self):
        """Check that gradient is all ones."""
        link = IdentityLink()
        dist = BinomialDist(levels=1)
        mu = np.array([1.0, 2.0, 3.0])
        grad = link.gradient(mu, dist)
        np.testing.assert_array_equal(grad, np.ones_like(mu))


class TestLogLink:
    """Tests for LogLink class."""

    def test_init(self):
        """Check that LogLink can be instantiated."""
        link = LogLink()
        assert link._name == "log"

    def test_link_mu_inverse(self):
        """Check that link and mu are inverses of each other."""
        link = LogLink()
        dist = BinomialDist(levels=1)
        mu_original = np.array([0.5, 1.0, 2.0, 5.0])
        lp = link.link(mu_original, dist)
        mu_recovered = link.mu(lp, dist)
        np.testing.assert_array_almost_equal(mu_original, mu_recovered)


class TestInverseLink:
    """Tests for InverseLink class."""

    def test_init(self):
        """Check that InverseLink can be instantiated."""
        link = InverseLink()
        assert link._name == "inverse"

    def test_link_mu_inverse(self):
        """Check that link and mu are inverses of each other."""
        link = InverseLink()
        dist = BinomialDist(levels=1)
        mu_original = np.array([0.5, 1.0, 2.0, 5.0])
        lp = link.link(mu_original, dist)
        mu_recovered = link.mu(lp, dist)
        np.testing.assert_array_almost_equal(mu_original, mu_recovered)


class TestInvSquaredLink:
    """Tests for InvSquaredLink class."""

    def test_init(self):
        """Check that InvSquaredLink can be instantiated."""
        link = InvSquaredLink()
        assert link._name == "inv_squared"

    def test_link_mu_inverse(self):
        """Check that link and mu are inverses of each other for positive mu."""
        link = InvSquaredLink()
        dist = BinomialDist(levels=1)
        mu_original = np.array([0.5, 1.0, 2.0, 5.0])
        lp = link.link(mu_original, dist)
        mu_recovered = link.mu(lp, dist)
        np.testing.assert_array_almost_equal(mu_original, mu_recovered)


class TestLinksDictionary:
    """Tests for LINKS dictionary."""

    def test_all_links_present(self):
        """Check that all expected links are in the dictionary."""
        expected_links = [
            "identity",
            "log",
            "logit",
            "probit",
            "inverse",
            "inv_squared",
        ]
        for link_name in expected_links:
            assert link_name in LINKS

    def test_links_are_callable(self):
        """Check that all links can be instantiated."""
        for link_name, link_class in LINKS.items():
            link = link_class()
            assert link._name == link_name
