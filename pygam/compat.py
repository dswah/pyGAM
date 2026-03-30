"""
pygam/compat.py
===============
Scikit-learn compatible wrappers for pyGAM estimators.

Addresses the long-standing clone/GridSearchCV/Pipeline incompatibility
documented in issues #168, #247, #291, and partially fixed for LogisticGAM
in #482. This module provides drop-in wrappers that satisfy sklearn's
BaseEstimator contract without modifying the core pyGAM classes.

Root cause
----------
pyGAM's GAM base class mutates constructor parameters during __init__
(notably `callbacks` and `terms`), which breaks sklearn's clone() contract:

    from sklearn.base import clone
    clone(LinearGAM())  # RuntimeError on unpatched pyGAM

These wrappers isolate mutable state, expose clean get_params/set_params,
and return `self` from fit() so all sklearn meta-estimators work correctly:

    GridSearchCV, cross_val_score, Pipeline, check_estimator

Usage
-----
    from pygam.compat import SklearnLinearGAM, SklearnLogisticGAM

    # Pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gam", SklearnLinearGAM(n_splines=20, lam=0.6)),
    ])
    pipe.fit(X_train, y_train)

    # GridSearchCV
    grid = GridSearchCV(
        SklearnLinearGAM(),
        param_grid={"n_splines": [10, 20, 25], "lam": [0.1, 1.0, 10.0]},
        cv=5,
        scoring="r2",
    )
    grid.fit(X, y)

    # cross_val_score
    scores = cross_val_score(SklearnLinearGAM(), X, y, cv=5)
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data


# ─────────────────────────────────────────────────────────────────────────────
# Base mixin
# ─────────────────────────────────────────────────────────────────────────────

class _GAMCompatMixin:
    """
    Internal mixin that isolates mutable pyGAM state from sklearn's clone().
    """

    def _make_gam(self):
        raise NotImplementedError

    def fit(self, X, y, sample_weight=None):
        X, y = validate_data(self, X, y)
        self.gam_ = self._make_gam()
        self.gam_.fit(X, y)
        self.is_fitted_ = True
        return self

    def _check_fitted(self):
        check_is_fitted(self, "gam_")

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags


# ─────────────────────────────────────────────────────────────────────────────
# LinearGAM wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SklearnLinearGAM(_GAMCompatMixin, RegressorMixin, BaseEstimator):
    """
    Scikit-learn compatible wrapper for pyGAM's LinearGAM.

    Parameters
    ----------
    n_splines : int, default=25
    lam : float, default=0.6
    max_iter : int, default=100
    tol : float, default=1e-4
    fit_intercept : bool, default=True
    """

    def __init__(
        self,
        n_splines: int = 25,
        lam: float = 0.6,
        max_iter: int = 100,
        tol: float = 1e-4,
        fit_intercept: bool = True,
    ):
        self.n_splines = n_splines
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def _make_gam(self):
        from pygam import LinearGAM
        return LinearGAM(
            n_splines=self.n_splines,
            lam=self.lam,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
        )

    def predict(self, X):
        check_is_fitted(self, "gam_")
        X = validate_data(self, X, reset=False)
        return self.gam_.predict(X)

    def predict_interval(self, X, width: float = 0.95):
        check_is_fitted(self, "gam_")
        X = validate_data(self, X, reset=False)
        return self.gam_.prediction_intervals(X, width=width)


# ─────────────────────────────────────────────────────────────────────────────
# LogisticGAM wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SklearnLogisticGAM(_GAMCompatMixin, ClassifierMixin, BaseEstimator):
    """
    Scikit-learn compatible wrapper for pyGAM's LogisticGAM.

    Parameters
    ----------
    n_splines : int, default=25
    lam : float, default=0.6
    max_iter : int, default=100
    tol : float, default=1e-4
    fit_intercept : bool, default=True
    """

    def __init__(
        self,
        n_splines: int = 25,
        lam: float = 0.6,
        max_iter: int = 100,
        tol: float = 1e-4,
        fit_intercept: bool = True,
    ):
        self.n_splines = n_splines
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def _make_gam(self):
        from pygam import LogisticGAM
        return LogisticGAM(
            n_splines=self.n_splines,
            lam=self.lam,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
        )

    def fit(self, X, y, sample_weight=None):
        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)
        self.gam_ = self._make_gam()
        self.gam_.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "gam_")
        X = validate_data(self, X, reset=False)
        return self.gam_.predict(X).astype(int)

    def predict_proba(self, X):
        check_is_fitted(self, "gam_")
        X = validate_data(self, X, reset=False)
        pos_prob = self.gam_.predict_proba(X)
        return np.column_stack([1 - pos_prob, pos_prob])

    def decision_function(self, X):
        check_is_fitted(self, "gam_")
        X = validate_data(self, X, reset=False)
        return self.gam_.predict_mu(X)


# ─────────────────────────────────────────────────────────────────────────────
# PoissonGAM wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SklearnPoissonGAM(_GAMCompatMixin, RegressorMixin, BaseEstimator):
    """
    Scikit-learn compatible wrapper for pyGAM's PoissonGAM.

    Parameters
    ----------
    n_splines : int, default=25
    lam : float, default=0.6
    max_iter : int, default=100
    tol : float, default=1e-4
    fit_intercept : bool, default=True
    """

    def __init__(
        self,
        n_splines: int = 25,
        lam: float = 0.6,
        max_iter: int = 100,
        tol: float = 1e-4,
        fit_intercept: bool = True,
    ):
        self.n_splines = n_splines
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def _make_gam(self):
        from pygam import PoissonGAM
        return PoissonGAM(
            n_splines=self.n_splines,
            lam=self.lam,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
        )

    def predict(self, X):
        check_is_fitted(self, "gam_")
        X = validate_data(self, X, reset=False)
        return self.gam_.predict(X)