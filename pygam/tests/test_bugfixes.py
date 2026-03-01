"""Tests for sklearn estimator contract bug fixes.

Each test reproduces the reported issue and verifies the fix.
"""

import numpy as np
import pytest

from pygam import GammaGAM, LinearGAM, LogisticGAM, PoissonGAM, f, l, s
from pygam.datasets import default, mcycle, wage
from pygam.terms import TermList

# ---------------------------------------------------------------------------
# #280  GAM.score() should work and guard against unfitted models
# ---------------------------------------------------------------------------


class TestScoreMethod:
    """Fix for GitHub issue #280."""

    def test_score_works_after_fit(self):
        X, y = mcycle(return_X_y=True)
        gam = LinearGAM().fit(X, y)
        score = gam.score(X, y)
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_score_raises_when_not_fitted(self):
        gam = LinearGAM()
        X, y = mcycle(return_X_y=True)
        with pytest.raises(AttributeError, match="GAM has not been fitted"):
            gam.score(X, y)

    def test_logistic_score_works(self):
        X, y = default(return_X_y=True)
        gam = LogisticGAM().fit(X, y)
        score = gam.score(X, y)
        assert isinstance(score, (float, np.floating))
        assert 0 <= score <= 1


# ---------------------------------------------------------------------------
# #283  predict_proba should return (n_samples, 2) array
# ---------------------------------------------------------------------------


class TestPredictProba:
    """Fix for GitHub issue #283."""

    def test_predict_proba_shape(self):
        X, y = default(return_X_y=True)
        gam = LogisticGAM().fit(X, y)
        proba = gam.predict_proba(X)
        assert proba.ndim == 2
        assert proba.shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self):
        X, y = default(return_X_y=True)
        gam = LogisticGAM().fit(X, y)
        proba = gam.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_predict_proba_columns_are_complementary(self):
        X, y = default(return_X_y=True)
        gam = LogisticGAM().fit(X, y)
        proba = gam.predict_proba(X)
        np.testing.assert_allclose(proba[:, 0] + proba[:, 1], 1.0)


# ---------------------------------------------------------------------------
# #340 / #333  clone() should preserve terms
# ---------------------------------------------------------------------------


class TestClonePreservesTerms:
    """Fix for GitHub issues #340 and #333."""

    def test_get_params_returns_original_terms(self):
        terms = s(0) + l(1) + f(2)
        gam = LinearGAM(terms=terms)
        params = gam.get_params()
        assert "terms" in params
        assert isinstance(params["terms"], TermList)

    def test_clone_roundtrip_preserves_terms(self):
        terms = s(0) + l(1) + f(2)
        gam = LinearGAM(terms=terms)
        params = gam.get_params()
        gam2 = LinearGAM(**params)
        assert isinstance(gam2.terms, TermList)
        assert len(gam2.terms) == len(terms)

    def test_clone_auto_terms_roundtrip(self):
        gam = LinearGAM()
        params = gam.get_params()
        assert params["terms"] == "auto"
        gam2 = LinearGAM(**params)
        assert gam2.terms == "auto"

    def test_fitted_model_clone_roundtrip(self):
        X, y = wage(return_X_y=True)
        terms = s(0) + s(1) + f(2)
        gam = LinearGAM(terms=terms).fit(X, y)
        params = gam.get_params()
        gam2 = LinearGAM(**params)
        # The cloned model should be fittable
        gam2.fit(X, y)
        assert gam2._is_fitted

    def test_sklearn_clone_compatible(self):
        """Test that sklearn.base.clone works if available."""
        try:
            from sklearn.base import clone
        except ImportError:
            pytest.skip("sklearn not installed")
        terms = s(0) + s(1) + f(2)
        gam = LinearGAM(terms=terms)
        cloned = clone(gam)
        assert isinstance(cloned.terms, TermList)
        # After cloning, the model should be fittable
        X, y = wage(return_X_y=True)
        cloned.fit(X, y)
        assert cloned._is_fitted


# ---------------------------------------------------------------------------
# #273  get_params should work on unfitted models without NoneType errors
# ---------------------------------------------------------------------------


class TestGetParamsUnfitted:
    """Fix for GitHub issue #273."""

    def test_unfitted_gam_get_params(self):
        gam = LinearGAM()
        params = gam.get_params()
        assert isinstance(params, dict)
        assert "terms" in params

    def test_unfitted_logistic_get_params(self):
        gam = LogisticGAM()
        params = gam.get_params()
        assert isinstance(params, dict)

    def test_unfitted_gam_with_terms_get_params(self):
        gam = LinearGAM(terms=s(0) + l(1))
        params = gam.get_params()
        assert isinstance(params, dict)
        assert isinstance(params["terms"], TermList)


# ---------------------------------------------------------------------------
# #422  sklearn v1.7+ __sklearn_tags__ compatibility
# ---------------------------------------------------------------------------


class TestSklearnTags:
    """Fix for GitHub issue #422."""

    def test_linear_gam_has_regressor_tags(self):
        pytest.importorskip("sklearn")
        tags = LinearGAM().__sklearn_tags__()
        assert tags.estimator_type == "regressor"
        assert tags.regressor_tags is not None

    def test_logistic_gam_has_classifier_tags(self):
        pytest.importorskip("sklearn")
        tags = LogisticGAM().__sklearn_tags__()
        assert tags.estimator_type == "classifier"
        assert tags.classifier_tags is not None

    def test_poisson_gam_has_regressor_tags(self):
        pytest.importorskip("sklearn")
        tags = PoissonGAM().__sklearn_tags__()
        assert tags.estimator_type == "regressor"

    def test_gamma_gam_has_regressor_tags(self):
        pytest.importorskip("sklearn")
        tags = GammaGAM().__sklearn_tags__()
        assert tags.estimator_type == "regressor"


# ---------------------------------------------------------------------------
# #247  LogisticGAM full sklearn GridSearchCV compatibility
# ---------------------------------------------------------------------------


class TestSklearnGridSearchCV:
    """Fix for GitHub issue #247."""

    def test_logistic_gam_has_classes(self):
        """LogisticGAM.fit() should set classes_ attribute."""
        X, y = default(return_X_y=True)
        gam = LogisticGAM().fit(X, y)
        np.testing.assert_array_equal(gam.classes_, [0, 1])

    def test_logistic_gam_predict_returns_int(self):
        """predict() should return int array, not bool."""
        X, y = default(return_X_y=True)
        gam = LogisticGAM().fit(X, y)
        preds = gam.predict(X)
        assert preds.dtype == np.int_ or np.issubdtype(preds.dtype, np.integer)

    def test_logistic_gam_decision_function(self):
        """decision_function() returns log-odds for each sample."""
        X, y = default(return_X_y=True)
        gam = LogisticGAM().fit(X, y)
        df = gam.decision_function(X)
        assert df.shape == (len(X),)
        assert np.all(np.isfinite(df))

    def test_decision_function_unfitted_raises(self):
        gam = LogisticGAM()
        X, _ = default(return_X_y=True)
        with pytest.raises(AttributeError, match="GAM has not been fitted"):
            gam.decision_function(X)

    def test_sklearn_gridsearchcv(self):
        """LogisticGAM should work inside sklearn GridSearchCV."""
        try:
            from sklearn.model_selection import GridSearchCV
        except ImportError:
            pytest.skip("sklearn not installed")
        np.random.seed(42)
        X = np.random.randn(200, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        gam = LogisticGAM(s(0) + s(1))
        cv = GridSearchCV(gam, param_grid={}, cv=3, scoring="accuracy")
        cv.fit(X, y)
        assert cv.best_score_ > 0.5

    def test_sklearn_cross_val_score(self):
        """LogisticGAM should work with cross_val_score."""
        try:
            from sklearn.model_selection import cross_val_score
        except ImportError:
            pytest.skip("sklearn not installed")
        np.random.seed(42)
        X = np.random.randn(200, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        gam = LogisticGAM(s(0) + s(1))
        scores = cross_val_score(gam, X, y, cv=3, scoring="accuracy")
        assert len(scores) == 3
        assert all(s > 0.4 for s in scores)
