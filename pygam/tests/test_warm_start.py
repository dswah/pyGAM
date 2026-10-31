from collections import defaultdict

import numpy as np
import pytest

from pygam import LinearGAM


def _n_iters(gam):
    """number of PIRLS iterations performed in the most recent fit"""
    return len(gam.logs_["diffs"])


def _refit(gam, X, y):
    """refit a model, resetting its logs so iterations can be counted"""
    gam.logs_ = defaultdict(list)
    return gam.fit(X, y)


def test_warm_start_default_is_false(mcycle_X_y):
    """
    warm_start should default to False, and be a model parameter
    """
    gam = LinearGAM()
    assert gam.warm_start is False
    assert gam.get_params()["warm_start"] is False


def test_warm_start_reuses_previous_coef(mcycle_X_y):
    """
    with warm_start=True, refitting should start from the existing coef_,
    so a refit seeded with converged coefficients converges immediately
    """
    X, y = mcycle_X_y

    gam = LinearGAM(warm_start=True).fit(X, y)
    coef = gam.coef_.copy()

    gam = _refit(gam, X, y)

    # starting from the solution should take a single iteration
    assert _n_iters(gam) == 1
    assert np.allclose(gam.coef_, coef)


def test_warm_start_false_ignores_previous_coef(mcycle_X_y):
    """
    with warm_start=False (default), refitting should ignore the existing
    coef_ and start from a fresh initial estimate, just like a new model
    """
    X, y = mcycle_X_y

    gam = LinearGAM(warm_start=False).fit(X, y)
    n_fresh = _n_iters(LinearGAM().fit(X, y))

    gam = _refit(gam, X, y)

    # the seeded coefficients were already converged, so reusing them
    # would converge immediately; a cold start takes the full fit
    assert _n_iters(gam) == n_fresh
    assert _n_iters(gam) > 1


def test_warm_start_false_ignores_seeded_coef(mcycle_X_y):
    """
    with warm_start=False, even a manually seeded coef_ should not
    affect the result of a fit
    """
    X, y = mcycle_X_y

    gam = LinearGAM(warm_start=False).fit(X, y)
    coef = gam.coef_.copy()

    # seed the model with very different coefficients
    gam.coef_ = coef + 100.0
    gam.fit(X, y)

    assert np.allclose(gam.coef_, coef)


def test_warm_start_must_be_bool(mcycle_X_y):
    """
    warm_start must be a bool
    """
    X, y = mcycle_X_y

    with pytest.raises(ValueError):
        LinearGAM(warm_start="yes").fit(X, y)


def test_gridsearch_warm_starts_regardless_of_param(mcycle_X_y):
    """
    gridsearch should keep warm starting from the previous candidate model,
    and should give the same results, whatever the model's warm_start is.
    the model's own warm_start setting should be unchanged afterwards.
    """
    X, y = mcycle_X_y
    lams = np.logspace(-3, 3, 5)

    gam_cold = LinearGAM(warm_start=False)
    scores_cold = gam_cold.gridsearch(X, y, lam=lams, return_scores=True)

    gam_warm = LinearGAM(warm_start=True)
    scores_warm = gam_warm.gridsearch(X, y, lam=lams, return_scores=True)

    assert list(scores_cold.values()) == list(scores_warm.values())
    assert np.allclose(gam_cold.coef_, gam_warm.coef_)

    # gridsearch must not change the model's own warm_start setting
    assert gam_cold.warm_start is False
    assert gam_warm.warm_start is True
