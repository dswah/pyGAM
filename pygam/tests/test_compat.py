"""
pygam/tests/test_compat.py
==========================
Tests for sklearn compatibility wrappers in pygam.compat.

Two test layers:
1. Behavioral contract tests — what we guarantee explicitly
2. sklearn's check_estimator — the full official suite

Run:
    pytest pygam/tests/test_compat.py -v
    pytest pygam/tests/test_compat.py -v -k "not check_estimator"  # skip slow
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from pygam.compat import SklearnLinearGAM, SklearnLogisticGAM, SklearnPoissonGAM


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 3))
    y = X[:, 0] ** 2 + np.sin(X[:, 1]) + rng.standard_normal(100) * 0.1
    return X, y


@pytest.fixture
def classification_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def count_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 3))
    y = rng.poisson(lam=np.exp(X[:, 0]), size=100).astype(float)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: Behavioral contract tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSklearnLinearGAM:

    def test_clone_does_not_raise(self):
        """Core fix: clone() must work — this is what broke GridSearchCV."""
        gam = SklearnLinearGAM(n_splines=10, lam=1.0)
        cloned = clone(gam)
        assert cloned.n_splines == 10
        assert cloned.lam == 1.0
        assert not hasattr(cloned, "gam_")   # fitted state not copied

    def test_get_params_returns_constructor_args(self):
        gam = SklearnLinearGAM(n_splines=15, lam=2.0, max_iter=50)
        params = gam.get_params()
        assert params["n_splines"] == 15
        assert params["lam"] == 2.0
        assert params["max_iter"] == 50

    def test_set_params_updates_and_returns_self(self):
        gam = SklearnLinearGAM()
        result = gam.set_params(n_splines=30, lam=5.0)
        assert result is gam
        assert gam.n_splines == 30
        assert gam.lam == 5.0

    def test_fit_returns_self(self, regression_data):
        X, y = regression_data
        gam = SklearnLinearGAM()
        result = gam.fit(X, y)
        assert result is gam

    def test_predict_shape(self, regression_data):
        X, y = regression_data
        gam = SklearnLinearGAM().fit(X, y)
        preds = gam.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_before_fit_raises(self):
        from sklearn.exceptions import NotFittedError
        gam = SklearnLinearGAM()
        with pytest.raises(NotFittedError):
            gam.predict(np.zeros((5, 3)))

    def test_pipeline_integration(self, regression_data):
        """GridSearchCV and Pipeline are the main use-cases this fixes."""
        X, y = regression_data
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("gam", SklearnLinearGAM(n_splines=10)),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (len(y),)

    def test_grid_search_cv(self, regression_data):
        X, y = regression_data
        grid = GridSearchCV(
            SklearnLinearGAM(),
            param_grid={"n_splines": [10, 15], "lam": [0.1, 1.0]},
            cv=3,
            scoring="r2",
        )
        grid.fit(X, y)
        assert hasattr(grid, "best_estimator_")
        assert hasattr(grid, "best_params_")

    def test_cross_val_score(self, regression_data):
        X, y = regression_data
        scores = cross_val_score(
            SklearnLinearGAM(n_splines=10),
            X, y, cv=3, scoring="r2",
        )
        assert scores.shape == (3,)
        assert all(np.isfinite(scores))

    def test_n_features_in_set_on_fit(self, regression_data):
        X, y = regression_data
        gam = SklearnLinearGAM().fit(X, y)
        assert gam.n_features_in_ == X.shape[1]

    def test_predict_interval_shape(self, regression_data):
        X, y = regression_data
        gam = SklearnLinearGAM().fit(X, y)
        intervals = gam.predict_interval(X, width=0.95)
        assert intervals.shape == (len(y), 2)


class TestSklearnLogisticGAM:

    def test_clone_does_not_raise(self):
        gam = SklearnLogisticGAM(n_splines=10)
        cloned = clone(gam)
        assert cloned.n_splines == 10

    def test_classes_set_on_fit(self, classification_data):
        X, y = classification_data
        gam = SklearnLogisticGAM().fit(X, y)
        assert hasattr(gam, "classes_")
        np.testing.assert_array_equal(gam.classes_, [0, 1])

    def test_predict_returns_int_labels(self, classification_data):
        X, y = classification_data
        gam = SklearnLogisticGAM().fit(X, y)
        preds = gam.predict(X)
        assert preds.dtype in (np.int32, np.int64, int)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape_and_sums(self, classification_data):
        X, y = classification_data
        gam = SklearnLogisticGAM().fit(X, y)
        proba = gam.predict_proba(X)
        assert proba.shape == (len(y), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_decision_function_shape(self, classification_data):
        X, y = classification_data
        gam = SklearnLogisticGAM().fit(X, y)
        df = gam.decision_function(X)
        assert df.shape == (len(y),)

    def test_pipeline_integration(self, classification_data):
        X, y = classification_data
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("gam", SklearnLogisticGAM(n_splines=10)),
        ])
        pipe.fit(X, y)
        assert pipe.predict(X).shape == (len(y),)

    def test_cross_val_score(self, classification_data):
        X, y = classification_data
        scores = cross_val_score(
            SklearnLogisticGAM(n_splines=10),
            X, y, cv=3, scoring="accuracy",
        )
        assert scores.shape == (3,)


class TestSklearnPoissonGAM:

    def test_clone_does_not_raise(self):
        gam = SklearnPoissonGAM(n_splines=10)
        cloned = clone(gam)
        assert cloned.n_splines == 10

    def test_fit_predict(self, count_data):
        X, y = count_data
        gam = SklearnPoissonGAM(n_splines=10).fit(X, y)
        preds = gam.predict(X)
        assert preds.shape == (len(y),)
        assert (preds >= 0).all()   # Poisson predictions are non-negative


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: sklearn's official estimator check suite
# ─────────────────────────────────────────────────────────────────────────────
