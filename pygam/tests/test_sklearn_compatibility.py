from __future__ import annotations

import inspect

import numpy as np
import pytest
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator, parametrize_with_checks

from pygam import GAM
from pygam.tests.utils.discovery import all_estimators

rng = np.random.default_rng(42)

GAM_estimators = [est for _, est in all_estimators()]


def generate_regression_data():
    # positive regression data to ensure compatibility with GammaGAM and InvGaussGAM
    X = np.abs(rng.standard_normal((100, 3)))
    y = np.abs(X[:, 0] ** 2 + np.sin(X[:, 1]) + rng.standard_normal(100) * 0.1)
    return X, y


def generate_classification_data():
    X = rng.standard_normal((100, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def count_data():
    X = rng.standard_normal((100, 3))
    y = rng.poisson(lam=np.exp(X[:, 0]), size=100).astype(float)
    return X, y


@pytest.mark.skip(reason="Further development work needed for full compliance")
@pytest.mark.parametrize("estimator", [el for _, el in all_estimators()])
def test_estimators_shallow(estimator):
    """
    check that the estimators are fully compatible with scikit-learn.
    """
    check_estimator(estimator())


@pytest.mark.skip(reason="Further development work needed for full compliance")
@parametrize_with_checks([est() for _, est in all_estimators()])
def test_estimators_full(estimator, check, request):
    """Check for full compliance with scikit-learn API"""
    check(estimator)


@pytest.mark.parametrize("estimator", GAM_estimators)
def test_GridSearchCV(estimator):
    """
    check that the sklearn GridSearchCV works with pyGAM's models
    """
    mro = inspect.getmro(estimator)
    if RegressorMixin in mro or estimator is GAM:
        X, y = generate_regression_data()
    elif ClassifierMixin in mro:
        X, y = generate_classification_data()
    else:
        print(estimator)
        print(mro)
        raise NotImplementedError(
            "Tests only implemented for regression and classification models."
        )

    grid_search = GridSearchCV(
        estimator(),
        param_grid={"n_splines": [10, 15], "lam": [0.1, 1.0]},
        cv=3,
        scoring="r2",
    )
    grid_search.fit(X, y)
    assert hasattr(grid_search, "best_estimator_")
    assert hasattr(grid_search, "best_params_")


@pytest.mark.parametrize("estimator", GAM_estimators)
def test_premature_predict_raises_NotFittedError(estimator):
    """
    check that the sklearn NotFittedError is correctly raised for improper predict/fit order
    """
    gam = estimator()
    with pytest.raises(NotFittedError):
        gam.predict(np.zeros((5, 3)))


@pytest.mark.parametrize("estimator", GAM_estimators)
def test_Pipeline(estimator):
    """
    check that the sklearn.Pipeline works with pyGAM's models
    """
    mro = inspect.getmro(estimator)
    if RegressorMixin in mro or estimator is GAM:
        X, y = generate_regression_data()
    elif ClassifierMixin in mro:
        X, y = generate_classification_data()
    else:
        raise NotImplementedError(
            "Tests only implemented for regression and classification models."
        )
    print(estimator.__name__)
    print(y.min(), y.max())
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("gam", estimator(n_splines=10))]
    )
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    assert y_pred.shape == (len(y),)
