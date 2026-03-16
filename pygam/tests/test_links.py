"""Tests for link functions – focusing on numerical stability of LogitLink."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from pygam.links import LogitLink, IdentityLink, LogLink


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dist(levels=1.0):
    """Return a lightweight mock that mimics dist.levels."""
    dist = MagicMock()
    dist.levels = levels
    return dist


# ---------------------------------------------------------------------------
# LogitLink.mu – numerical stability (issue #534)
# ---------------------------------------------------------------------------

class TestLogitLinkMuStability:
    """LogitLink.mu must never return NaN or Inf regardless of lp magnitude."""

    def test_mu_no_nan_for_large_positive_lp(self):
        """lp > 709 used to overflow np.exp → NaN; expit handles it correctly."""
        dist = _make_dist(levels=1.0)
        link = LogitLink()
        large_lp = np.array([500.0, 710.0, 1_000.0, 1e6])
        mu = link.mu(large_lp, dist)
        assert not np.any(np.isnan(mu)), "NaN detected for large positive lp"
        assert not np.any(np.isinf(mu)), "Inf detected for large positive lp"
        # For very large lp the sigmoid saturates to 1.0
        np.testing.assert_allclose(mu, dist.levels, atol=1e-6)

    def test_mu_no_nan_for_large_negative_lp(self):
        """lp << 0 should saturate to 0, not NaN."""
        dist = _make_dist(levels=1.0)
        link = LogitLink()
        large_neg_lp = np.array([-500.0, -710.0, -1_000.0, -1e6])
        mu = link.mu(large_neg_lp, dist)
        assert not np.any(np.isnan(mu)), "NaN detected for large negative lp"
        assert not np.any(np.isinf(mu)), "Inf detected for large negative lp"
        np.testing.assert_allclose(mu, 0.0, atol=1e-6)

    def test_mu_midpoint(self):
        """lp = 0 → mu = levels/2 (sigmoid(0) = 0.5)."""
        dist = _make_dist(levels=1.0)
        link = LogitLink()
        mu = link.mu(np.array([0.0]), dist)
        np.testing.assert_allclose(mu, 0.5, atol=1e-10)

    def test_mu_respects_levels(self):
        """When levels != 1, the output should be scaled accordingly."""
        levels = 10
        dist = _make_dist(levels=levels)
        link = LogitLink()
        # sigmoid(0) = 0.5, so mu should be levels * 0.5
        mu = link.mu(np.array([0.0]), dist)
        np.testing.assert_allclose(mu, levels * 0.5, atol=1e-10)

        # Large lp saturates to levels
        mu_large = link.mu(np.array([1e6]), dist)
        np.testing.assert_allclose(mu_large, levels, atol=1e-6)

    def test_mu_is_inverse_of_link(self):
        """mu(link(p)) should round-trip back to p for normal probability values."""
        dist = _make_dist(levels=1.0)
        link = LogitLink()
        p = np.linspace(0.01, 0.99, 50)
        lp = link.link(p, dist)
        p_recovered = link.mu(lp, dist)
        np.testing.assert_allclose(p_recovered, p, atol=1e-12)

    def test_mu_output_in_valid_range(self):
        """All outputs should lie strictly in [0, levels] for any finite lp."""
        dist = _make_dist(levels=1.0)
        link = LogitLink()
        lp = np.linspace(-1000, 1000, 2001)
        mu = link.mu(lp, dist)
        assert np.all(mu >= 0.0), "mu below 0"
        assert np.all(mu <= dist.levels), "mu above levels"


# ---------------------------------------------------------------------------
# LogitLink.gradient – soft-clipping prevents division-by-zero
# ---------------------------------------------------------------------------

class TestLogitLinkGradientStability:
    """LogitLink.gradient must be finite for all mu in [0, levels]."""

    def test_gradient_finite_at_zero(self):
        """mu = 0 should not produce inf/NaN gradient."""
        dist = _make_dist(levels=1.0)
        link = LogitLink()
        grad = link.gradient(np.array([0.0]), dist)
        assert np.all(np.isfinite(grad)), f"Non-finite gradient at mu=0: {grad}"

    def test_gradient_finite_at_levels(self):
        """mu = dist.levels should not produce inf/NaN gradient."""
        dist = _make_dist(levels=1.0)
        link = LogitLink()
        grad = link.gradient(np.array([dist.levels]), dist)
        assert np.all(np.isfinite(grad)), f"Non-finite gradient at mu=levels: {grad}"

    def test_gradient_positive(self):
        """The logit link is a monotone increasing function so gradient should be +."""
        dist = _make_dist(levels=1.0)
        link = LogitLink()
        mu = np.linspace(0.01, 0.99, 50)
        grad = link.gradient(mu, dist)
        assert np.all(grad > 0), "Gradient should be positive"

    def test_gradient_finite_everywhere(self):
        """Gradient should be finite for all mu in [0, levels]."""
        dist = _make_dist(levels=1.0)
        link = LogitLink()
        mu = np.linspace(0.0, 1.0, 201)  # includes 0 and 1
        grad = link.gradient(mu, dist)
        assert np.all(np.isfinite(grad)), "Non-finite gradients detected"


# ---------------------------------------------------------------------------
# LogisticGAM integration test: fitting on perfectly-separated data
# ---------------------------------------------------------------------------

class TestLogisticGAMWithExtremeData:
    """LogisticGAM must not raise OptimizationError or produce NaN predictions
    when dealing with highly separable or extreme predictor values."""

    def test_predict_proba_no_nan_on_extreme_predictors(self):
        """Predict on extreme X values should never return NaN."""
        from pygam import LogisticGAM
        import numpy as np

        np.random.seed(42)
        # Build a well-separated binary dataset
        X_neg = np.random.randn(50, 1) - 20   # class 0, far left
        X_pos = np.random.randn(50, 1) + 20   # class 1, far right
        X = np.vstack([X_neg, X_pos])
        y = np.array([0] * 50 + [1] * 50)

        gam = LogisticGAM().fit(X, y)
        proba = gam.predict_proba(X)

        assert not np.any(np.isnan(proba)), "NaN probabilities from logistic GAM"
        assert np.all((proba >= 0) & (proba <= 1)), "Probabilities out of [0, 1]"

    def test_predict_proba_no_nan_for_very_large_linear_predictor(self):
        """Direct call to LogitLink.mu with extreme values must not give NaN,
        matching what happens inside PIRLS with poorly-scaled data."""
        from pygam.links import LogitLink

        link = LogitLink()
        dist = _make_dist(levels=1.0)

        extreme_lp = np.array([-1e10, -1000.0, -710.0, 0.0, 710.0, 1000.0, 1e10])
        mu = link.mu(extreme_lp, dist)

        assert not np.any(np.isnan(mu)), f"Got NaN in mu: {mu}"
        assert not np.any(np.isinf(mu)), f"Got Inf in mu: {mu}"
        np.testing.assert_array_less(-1e-15, mu)           # mu >= 0
        np.testing.assert_array_less(mu, dist.levels + 1e-15)  # mu <= levels
