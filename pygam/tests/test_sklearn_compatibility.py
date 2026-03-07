"""Tests for scikit-learn compatibility (clone, Pipeline, etc.)."""

import numpy as np
from sklearn.base import clone

from pygam import LinearGAM, LogisticGAM, s


class TestSklearnClone:
    """Regression tests for GitHub issue #478: clone() drops terms."""

    def test_clone_preserves_explicit_terms(self):
        """Cloning a GAM with explicit terms must preserve them."""
        gam = LinearGAM(s(0) + s(1))
        cloned = clone(gam)
        assert str(cloned.terms) == str(gam.terms)

    def test_clone_preserves_default_terms(self):
        """Cloning a GAM with default terms='auto' must still work."""
        gam = LinearGAM()
        cloned = clone(gam)
        assert cloned.terms == gam.terms

    def test_clone_preserves_terms_after_fit(self):
        """Cloning a fitted GAM must preserve the original constructor terms."""
        np.random.seed(0)
        X = np.random.randn(100, 2)
        y = np.random.randn(100)

        gam = LinearGAM(s(0) + s(1)).fit(X, y)
        cloned = clone(gam)

        # The original terms before fit were s(0) + s(1).
        # After fit, an intercept is appended: s(0) + s(1) + intercept.
        # clone() should preserve whatever get_params()["terms"] returns.
        assert str(cloned.terms) == str(gam.terms)

    def test_cloned_model_can_refit_with_same_results(self):
        """A cloned (unfitted) model with explicit terms must refit correctly."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = np.random.randn(100)

        gam = LinearGAM(s(0) + s(1))
        cloned = clone(gam)

        gam.fit(X, y)
        cloned.fit(X, y)

        # Both should produce the same predictions since they have
        # identical terms and are fitted on the same data.
        assert np.allclose(gam.predict(X), cloned.predict(X))

    def test_clone_logistic_gam_with_explicit_terms(self):
        """Cloning a LogisticGAM with explicit terms must preserve them."""
        gam = LogisticGAM(s(0) + s(1))
        cloned = clone(gam)
        assert str(cloned.terms) == str(gam.terms)
