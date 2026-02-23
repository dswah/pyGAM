"""
Tests to verify overflow fixes in links.py and pygam.py

These tests ensure that:
1. Extreme values don't cause overflow errors
2. Prediction quality remains adequate after clipping
"""

import numpy as np
import pytest
from pygam import LogisticGAM, PoissonGAM
from pygam.links import LogitLink, LogLink
from pygam.distributions import BinomialDist, PoissonDist


class TestLinkOverflow:
    """Test that link functions handle extreme values without overflow."""

    def test_logit_link_extreme_positive(self):
        """LogitLink.mu should not overflow with large positive values."""
        link = LogitLink()
        dist = BinomialDist(levels=1)
        lp = np.array([1000, 5000, 10000])
        # Should not raise and should return values close to 1
        mu = link.mu(lp, dist)
        assert np.all(np.isfinite(mu))
        assert np.allclose(mu, 1.0)

    def test_logit_link_extreme_negative(self):
        """LogitLink.mu should not underflow with large negative values."""
        link = LogitLink()
        dist = BinomialDist(levels=1)
        lp = np.array([-1000, -5000, -10000])
        # Should not raise and should return values close to 0
        mu = link.mu(lp, dist)
        assert np.all(np.isfinite(mu))
        assert np.allclose(mu, 0.0)

    def test_log_link_extreme_positive(self):
        """LogLink.mu should not overflow with large positive values."""
        link = LogLink()
        dist = PoissonDist()
        lp = np.array([1000, 5000, 10000])
        # Should not raise, returns very large but finite values
        mu = link.mu(lp, dist)
        assert np.all(np.isfinite(mu))
        # Should be capped at exp(700)
        assert np.all(mu == np.exp(700))

    def test_log_link_extreme_negative(self):
        """LogLink.mu should not underflow with large negative values."""
        link = LogLink()
        dist = PoissonDist()
        lp = np.array([-1000, -5000, -10000])
        # Should not raise, returns very small but finite values
        mu = link.mu(lp, dist)
        assert np.all(np.isfinite(mu))
        # Should be capped at exp(-700)
        assert np.all(mu == np.exp(-700))

    def test_logit_link_normal_range_unchanged(self):
        """LogitLink.mu should behave normally for typical values."""
        link = LogitLink()
        dist = BinomialDist(levels=1)
        lp = np.array([-5, -1, 0, 1, 5])
        mu = link.mu(lp, dist)
        expected = 1.0 / (1.0 + np.exp(-lp))
        assert np.allclose(mu, expected)

    def test_log_link_normal_range_unchanged(self):
        """LogLink.mu should behave normally for typical values."""
        link = LogLink()
        dist = PoissonDist()
        lp = np.array([-5, -1, 0, 1, 5])
        mu = link.mu(lp, dist)
        expected = np.exp(lp)
        assert np.allclose(mu, expected)


class TestGradientOverflow:
    """Test that gradient computations don't overflow in _W method."""

    def test_logistic_gam_extreme_predictions(self):
        """LogisticGAM should handle edge cases without FloatingPointError."""
        np.random.seed(42)
        # Create data that might push predictions to extremes
        X = np.random.randn(100, 1) * 10
        y = (X[:, 0] > 0).astype(int)

        gam = LogisticGAM()
        # Should not raise FloatingPointError
        gam.fit(X, y)

        # Predictions should be finite
        predictions = gam.predict_proba(X)
        assert np.all(np.isfinite(predictions))

    def test_poisson_gam_extreme_predictions(self):
        """PoissonGAM should handle edge cases without FloatingPointError."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = np.random.poisson(lam=5, size=100)

        gam = PoissonGAM()
        # Should not raise FloatingPointError
        gam.fit(X, y)

        # Predictions should be finite
        predictions = gam.predict(X)
        assert np.all(np.isfinite(predictions))


class TestPredictionQuality:
    """Test that clipping doesn't degrade prediction quality."""

    def test_logistic_gam_accuracy(self):
        """LogisticGAM should maintain good accuracy after overflow fixes."""
        np.random.seed(42)
        # Create linearly separable data
        X = np.random.randn(200, 1)
        y = (X[:, 0] > 0).astype(int)

        gam = LogisticGAM()
        gam.fit(X, y)
        predictions = gam.predict(X)

        accuracy = np.mean(predictions == y)
        # Should achieve high accuracy on linearly separable data
        assert accuracy > 0.9

    def test_logistic_gam_probability_calibration(self):
        """LogisticGAM probabilities should be well-calibrated."""
        np.random.seed(42)
        X = np.linspace(-3, 3, 100).reshape(-1, 1)
        # True probability follows logistic curve
        true_prob = 1 / (1 + np.exp(-2 * X[:, 0]))
        y = (np.random.rand(100) < true_prob).astype(int)

        gam = LogisticGAM()
        gam.fit(X, y)
        pred_prob = gam.predict_proba(X)

        # Probabilities should be between 0 and 1
        assert np.all(pred_prob >= 0)
        assert np.all(pred_prob <= 1)

        # Predicted probabilities should correlate with true probabilities
        correlation = np.corrcoef(true_prob, pred_prob)[0, 1]
        assert correlation > 0.8

    def test_poisson_gam_prediction_quality(self):
        """PoissonGAM should maintain prediction quality after overflow fixes."""
        np.random.seed(42)
        X = np.linspace(0, 5, 100).reshape(-1, 1)
        # True rate increases with X
        true_rate = np.exp(0.5 * X[:, 0])
        y = np.random.poisson(lam=true_rate.flatten())

        gam = PoissonGAM()
        gam.fit(X, y)
        predictions = gam.predict(X)

        # Predictions should be positive
        assert np.all(predictions > 0)

        # Predictions should correlate with true rate
        correlation = np.corrcoef(true_rate.flatten(), predictions)[0, 1]
        assert correlation > 0.8


class TestClipBoundaryValue:
    """Test that 700 is an appropriate clipping boundary."""

    def test_exp_700_is_finite(self):
        """np.exp(700) should be finite."""
        assert np.isfinite(np.exp(700))

    def test_exp_710_overflows(self):
        """np.exp(710) overflows to inf, justifying 700 as safe boundary."""
        assert np.isinf(np.exp(710))

    def test_exp_negative_700_is_finite(self):
        """np.exp(-700) should be finite (very small but not zero)."""
        result = np.exp(-700)
        assert np.isfinite(result)
        assert result > 0

    def test_safety_margin(self):
        """700 provides adequate safety margin below overflow threshold."""
        # Find approximate overflow threshold
        threshold = 709  # np.exp(709) is finite, np.exp(710) is inf
        margin = threshold - 700
        # At least 9 units of margin
        assert margin >= 9
