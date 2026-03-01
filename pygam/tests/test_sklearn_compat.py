"""Tests for scikit-learn interface compatibility.

Validates that pyGAM's Core.get_params / set_params behave according to the
scikit-learn estimator contract, allowing downstream integration with tools
like Pipeline, GridSearchCV, and clone().
"""

import inspect

import numpy as np
import pytest

from pygam import GAM, ExpectileGAM, LinearGAM, LogisticGAM, PoissonGAM
from pygam.core import Core
from pygam.terms import Intercept, SplineTerm

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _init_param_names(cls):
    """Extract explicit __init__ parameter names (excluding self / **kwargs)."""
    sig = inspect.signature(cls.__init__)
    return [
        name
        for name, p in sig.parameters.items()
        if name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    ]


# ---------------------------------------------------------------------------
# _get_param_names
# ---------------------------------------------------------------------------


class TestGetParamNames:
    """Tests for Core._get_param_names classmethod."""

    def test_core_returns_init_params(self):
        names = Core._get_param_names()
        assert "name" in names
        assert "line_width" in names
        assert "line_offset" in names

    def test_gam_includes_all_init_params(self):
        names = GAM._get_param_names()
        for p in _init_param_names(GAM):
            assert p in names, f"{p} missing from GAM._get_param_names"

    def test_linear_gam_includes_own_and_parent_params(self):
        names = LinearGAM._get_param_names()
        # LinearGAM's own param
        assert "scale" in names
        # inherited from GAM
        assert "max_iter" in names
        assert "tol" in names

    def test_spline_term_includes_all_init_params(self):
        names = SplineTerm._get_param_names()
        for p in _init_param_names(SplineTerm):
            assert p in names


# ---------------------------------------------------------------------------
# get_params - unfitted models
# ---------------------------------------------------------------------------


class TestGetParamsUnfitted:
    """get_params on freshly constructed objects."""

    def test_linear_gam_excludes_distribution_and_link(self):
        params = LinearGAM().get_params()
        assert "distribution" not in params
        assert "link" not in params

    def test_linear_gam_contains_own_params(self):
        params = LinearGAM().get_params()
        assert "scale" in params
        assert "max_iter" in params
        assert "callbacks" in params

    def test_logistic_gam_excludes_distribution_and_link(self):
        params = LogisticGAM().get_params()
        assert "distribution" not in params
        assert "link" not in params

    def test_gam_with_lam_kwarg(self):
        gam = GAM(lam=5)
        params = gam.get_params()
        assert params["lam"] == 5

    def test_core_params_do_not_leak(self):
        params = LinearGAM().get_params()
        for hidden in ("name", "line_width", "line_offset"):
            assert hidden not in params

    def test_private_attrs_do_not_leak(self):
        params = LinearGAM().get_params()
        for k in params:
            assert not k.startswith("_"), f"private attr {k} leaked"
            assert not k.endswith("_"), f"trailing-underscore attr {k} leaked"

    def test_term_get_params(self):
        t = SplineTerm(0, lam=3, n_splines=15)
        params = t.get_params()
        # lam is normalized to a list during validation
        assert np.allclose(params["lam"], [3])
        assert params["n_splines"] == 15
        assert params["feature"] == 0

    def test_intercept_get_params(self):
        # Intercept excludes many standard Term params
        params = Intercept().get_params()
        assert "feature" not in params


# ---------------------------------------------------------------------------
# get_params - fitted models
# ---------------------------------------------------------------------------


class TestGetParamsFitted:
    """Fitted attributes must never appear in get_params."""

    def test_fitted_attrs_absent(self, mcycle_X_y):
        X, y = mcycle_X_y
        gam = LinearGAM().fit(X, y)
        params = gam.get_params()
        for attr in ("coef_", "statistics_", "logs_"):
            assert attr not in params, f"{attr} leaked after fitting"

    def test_params_unchanged_after_fitting(self, mcycle_X_y):
        X, y = mcycle_X_y
        gam = LinearGAM(max_iter=50, tol=1e-3)
        before = gam.get_params()
        gam.fit(X, y)
        after = gam.get_params()
        # same keys
        assert set(before) == set(after)
        # scalar values preserved
        assert after["max_iter"] == 50
        assert after["tol"] == 1e-3


# ---------------------------------------------------------------------------
# set_params
# ---------------------------------------------------------------------------


class TestSetParams:
    """Verify set_params round-trips correctly with get_params."""

    def test_set_known_param(self):
        gam = GAM(lam=1)
        gam.set_params(lam=420)
        assert gam.lam == 420

    def test_set_unknown_param_no_effect(self):
        gam = GAM()
        gam.set_params(bogus=999)
        assert not hasattr(gam, "bogus")

    def test_force_sets_arbitrary_attr(self):
        gam = GAM()
        gam.set_params(force=True, bogus=999)
        assert gam.bogus == 999

    def test_roundtrip_get_set(self, mcycle_X_y):
        X, y = mcycle_X_y
        gam = LinearGAM(max_iter=75, scale=0.5)
        snapshot = gam.get_params()
        gam2 = LinearGAM()
        gam2.set_params(**snapshot)
        assert gam2.get_params() == snapshot


# ---------------------------------------------------------------------------
# repr stability
# ---------------------------------------------------------------------------


class TestRepr:
    """Ensure repr still renders cleanly after the get_params rewrite."""

    @pytest.mark.parametrize("cls", [LinearGAM, LogisticGAM, PoissonGAM, ExpectileGAM])
    def test_repr_parseable(self, cls):
        r = repr(cls())
        assert cls.__name__ in r
        assert r.endswith(")")

    def test_repr_does_not_contain_private_keys(self):
        r = repr(LinearGAM())
        assert "_constraint" not in r
        assert "_term_location" not in r


# ---------------------------------------------------------------------------
# gridsearch integration
# ---------------------------------------------------------------------------


class TestGridsearchCompat:
    """Gridsearch relies heavily on get_params / set_params internally."""

    def test_gridsearch_completes(self, mcycle_X_y):
        X, y = mcycle_X_y
        gam = LinearGAM()
        gam.gridsearch(X, y, lam=np.logspace(-2, 2, 3))
        assert gam._is_fitted
