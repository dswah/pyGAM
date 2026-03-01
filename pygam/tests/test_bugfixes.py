"""Tests for numerical stability bug fixes.

Each test reproduces the reported issue and verifies the fix.
"""

import warnings

import numpy as np
import pytest

from pygam import LinearGAM, LogisticGAM, s
from pygam.datasets import default, mcycle

# ---------------------------------------------------------------------------
# #367  Overflow RuntimeWarning in LogitLink and LogLink
# ---------------------------------------------------------------------------


class TestLinkOverflow:
    """Fix for GitHub issue #367."""

    def test_logit_mu_no_overflow_large_lp(self):
        """LogitLink.mu should not overflow for large positive linear predictor."""
        from pygam.distributions import BinomialDist
        from pygam.links import LogitLink

        link = LogitLink()
        dist = BinomialDist(levels=1)
        lp = np.array([-1000.0, -100.0, 0.0, 100.0, 1000.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mu = link.mu(lp, dist)
        assert np.all(np.isfinite(mu))
        assert mu[0] == pytest.approx(0.0, abs=1e-10)
        assert mu[-1] == pytest.approx(1.0, abs=1e-10)

    def test_log_mu_no_overflow(self):
        """LogLink.mu should not overflow for large linear predictor."""
        from pygam.distributions import PoissonDist
        from pygam.links import LogLink

        link = LogLink()
        dist = PoissonDist()
        lp = np.array([-800.0, 0.0, 700.0, 800.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mu = link.mu(lp, dist)
        assert np.all(np.isfinite(mu))

    def test_logit_gradient_no_division_by_zero(self):
        """LogitLink.gradient should not produce inf at boundary mu values."""
        from pygam.distributions import BinomialDist
        from pygam.links import LogitLink

        link = LogitLink()
        dist = BinomialDist(levels=1)
        mu = np.array([0.0, 0.5, 1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            grad = link.gradient(mu, dist)
        assert np.all(np.isfinite(grad))

    def test_logistic_gam_fit_no_overflow_warning(self):
        """LogisticGAM fitting should not produce overflow warnings."""
        np.random.seed(42)
        X = np.random.randn(200, 1)
        y = (X.ravel() > 0).astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            gam = LogisticGAM(s(0)).fit(X, y)
        assert gam._is_fitted


# ---------------------------------------------------------------------------
# #457  NormalDist.log_pdf divides scale by weights instead of sqrt(weights)
# ---------------------------------------------------------------------------


class TestNormalDistLogPdf:
    """Fix for GitHub issue #457."""

    def test_log_pdf_weights_use_sqrt(self):
        """With weight w, the effective SD should be scale/sqrt(w)."""
        from pygam.distributions import NormalDist

        dist = NormalDist(scale=2.0)
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.0, 2.0, 3.0])
        weights = np.array([4.0, 4.0, 4.0])
        log_pdf = dist.log_pdf(y, mu, weights)
        # At y == mu the log_pdf = -0.5 * log(2*pi*sigma_eff^2)
        # sigma_eff = 2.0 / sqrt(4.0) = 1.0
        import scipy.stats

        expected = scipy.stats.norm.logpdf(y, loc=mu, scale=1.0)
        np.testing.assert_allclose(log_pdf, expected)

    def test_log_pdf_unit_weights_unchanged(self):
        """With unit weights, log_pdf should use the original scale."""
        from pygam.distributions import NormalDist

        dist = NormalDist(scale=3.0)
        y = np.array([0.0, 1.0])
        mu = np.array([0.0, 0.0])
        log_pdf_no_w = dist.log_pdf(y, mu)
        log_pdf_w1 = dist.log_pdf(y, mu, weights=np.ones(2))
        np.testing.assert_allclose(log_pdf_no_w, log_pdf_w1)


# ---------------------------------------------------------------------------
# Bitwise NOT bug in GCV/UBRE estimator
# ---------------------------------------------------------------------------


class TestGCVUBRE:
    """Fix for bitwise NOT (~bool) in _estimate_GCV_UBRE."""

    def test_ubre_add_scale_true(self):
        """With add_scale=True, UBRE should not subtract the scale."""
        X, y = default(return_X_y=True)
        gam = LogisticGAM(s(0)).fit(X, y)
        _, ubre_with = gam._estimate_GCV_UBRE(X, y, add_scale=True)
        _, ubre_without = gam._estimate_GCV_UBRE(X, y, add_scale=False)
        assert ubre_with is not None
        assert ubre_without is not None
        # With add_scale=True the scale term is NOT subtracted => higher value
        assert ubre_with > ubre_without


# ---------------------------------------------------------------------------
# BinomialDist.log_pdf ignoring weights
# ---------------------------------------------------------------------------


class TestBinomialWeights:
    """Fix for BinomialDist.log_pdf silently ignoring sample weights."""

    def test_log_pdf_respects_weights(self):
        """Weighted log-pdf should differ from unweighted."""
        from pygam.distributions import BinomialDist

        dist = BinomialDist(levels=1)
        y = np.array([1.0, 0.0, 1.0, 0.0])
        mu = np.array([0.7, 0.3, 0.6, 0.4])
        unweighted = dist.log_pdf(y, mu)
        weights = np.array([2.0, 2.0, 0.5, 0.5])
        weighted = dist.log_pdf(y, mu, weights=weights)
        # Weighted should scale each element by its weight
        np.testing.assert_allclose(weighted, weights * unweighted)

    def test_unit_weights_equal_unweighted(self):
        """Weights of 1.0 should produce identical results to no weights."""
        from pygam.distributions import BinomialDist

        dist = BinomialDist(levels=1)
        y = np.array([1.0, 0.0, 1.0])
        mu = np.array([0.8, 0.2, 0.6])
        no_weights = dist.log_pdf(y, mu)
        unit_weights = dist.log_pdf(y, mu, weights=np.ones(3))
        np.testing.assert_allclose(no_weights, unit_weights)


# ---------------------------------------------------------------------------
# b_spline_basis extrapolation: bitwise NOT on bool+bool integer array
# ---------------------------------------------------------------------------


class TestBSplineBasisExtrapolation:
    """Fix for bitwise NOT on integer array in b_spline_basis.

    Using + on boolean arrays produces an integer array, then ~ gives
    bitwise NOT on ints (~0=-1, ~1=-2) instead of logical NOT, corrupting
    the basis for the first few interpolating data points.
    """

    def test_extrapolation_does_not_corrupt_interpolating_points(self):
        """Basis values for in-range points should be unaffected by
        out-of-range points in the same batch."""
        from pygam.utils import b_spline_basis

        edge_knots = np.array([0.0, 1.0])
        # In-range point at 0.5
        bases_alone = b_spline_basis(
            np.array([0.5]),
            edge_knots=edge_knots,
            n_splines=10,
            spline_order=3,
            sparse=False,
        )
        # Same point mixed with out-of-range points
        bases_mixed = b_spline_basis(
            np.array([0.5, -0.5, 1.5]),
            edge_knots=edge_knots,
            n_splines=10,
            spline_order=3,
            sparse=False,
        )
        # Row 0 (x=0.5) should be identical in both cases
        np.testing.assert_allclose(bases_alone[0], bases_mixed[0], atol=1e-14)

    def test_predict_unaffected_by_extrapolating_neighbors(self):
        """Predictions for in-range points should not change when
        out-of-range points are added to the prediction batch."""
        X, y = mcycle(return_X_y=True)
        gam = LinearGAM().fit(X, y)
        x_min, x_max = X.min(), X.max()
        x_mid = np.array([(x_min + x_max) / 2])
        pred_alone = gam.predict(x_mid)
        # Add extrapolating points around the in-range point
        x_with_extrap = np.array([x_mid[0], x_min - 10, x_max + 10])
        pred_mixed = gam.predict(x_with_extrap)
        np.testing.assert_allclose(pred_alone[0], pred_mixed[0], atol=1e-10)
