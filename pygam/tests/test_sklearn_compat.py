"""Tests for sklearn compatibility."""

import numpy as np
import pytest

from pygam import GAM, LinearGAM, LogisticGAM, PoissonGAM


def test_gam_has_sklearn_tags():
    """Test that GAM has __sklearn_tags__ method for sklearn 1.7+ compatibility."""
    gam = LinearGAM()
    assert hasattr(gam, "__sklearn_tags__"), "GAM should have __sklearn_tags__ method"


def test_randomized_search_cv_works():
    """Test that RandomizedSearchCV works with GAM models."""
    try:
        from sklearn.model_selection import RandomizedSearchCV
    except ImportError:
        pytest.skip("sklearn not installed")

    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = np.random.randn(100)

    gam = LinearGAM()
    param_dist = {"lam": [0.1, 0.5, 1.0]}

    rs = RandomizedSearchCV(gam, param_dist, n_iter=2, cv=2, random_state=42)
    rs.fit(X, y)

    assert rs.best_params_ is not None
    assert "lam" in rs.best_params_


def test_grid_search_cv_works():
    """Test that GridSearchCV works with GAM models."""
    try:
        from sklearn.model_selection import GridSearchCV
    except ImportError:
        pytest.skip("sklearn not installed")

    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = np.random.randn(50)

    gam = LinearGAM()
    param_grid = {"lam": [0.1, 1.0]}

    gs = GridSearchCV(gam, param_grid, cv=2)
    gs.fit(X, y)

    assert gs.best_params_ is not None
    assert "lam" in gs.best_params_


def test_all_gam_types_have_sklearn_tags():
    """Test that all GAM types have __sklearn_tags__ method."""
    gam_types = [GAM, LinearGAM, LogisticGAM, PoissonGAM]

    for gam_class in gam_types:
        gam = gam_class()
        assert hasattr(
            gam, "__sklearn_tags__"
        ), f"{gam_class.__name__} should have __sklearn_tags__ method"
