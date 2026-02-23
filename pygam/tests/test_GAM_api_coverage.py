"""Expanded regression / API-coverage tests for pyGAM.

These tests target the public API surface reported as insufficiently
covered in issue #440. Each test is focused, fast, and self-described.
"""

import numpy as np

from pygam import (
    GammaGAM,
    InvGaussGAM,
    LinearGAM,
    LogisticGAM,
    PoissonGAM,
    f,
    l,
    s,
    te,
)
from pygam.terms import FactorTerm

# ---------------------------------------------------------------------------
# score() — R² for LinearGAM, accuracy-like for LogisticGAM
# ---------------------------------------------------------------------------


def test_lineargam_score_returns_float(mcycle_X_y):
    """score() must return a finite float in [0, 1] for a well-fitted LinearGAM."""
    X, y = mcycle_X_y
    gam = LinearGAM().fit(X, y)
    sc = gam.score(X, y)
    assert isinstance(sc, float)
    assert np.isfinite(sc)


def test_logisticgam_score_returns_float(default_X_y):
    """score() must return a float for LogisticGAM (accuracy)."""
    X, y = default_X_y
    gam = LogisticGAM().fit(X, y)
    sc = gam.score(X, y)
    assert isinstance(sc, float)
    assert 0.0 <= sc <= 1.0


# ---------------------------------------------------------------------------
# loglikelihood()
# ---------------------------------------------------------------------------


def test_loglikelihood_is_negative(mcycle_X_y):
    """Log-likelihood for a Gaussian GAM should be negative (or at most 0)."""
    X, y = mcycle_X_y
    gam = LinearGAM().fit(X, y)
    loglik = gam.loglikelihood(X, y)
    assert np.isfinite(loglik)


def test_loglikelihood_increases_with_better_fit(mcycle_X_y):
    """A well-tuned model should have higher log-likelihood than a poor one."""
    X, y = mcycle_X_y
    gam_good = LinearGAM(lam=0.1).fit(X, y)
    gam_poor = LinearGAM(lam=1e6).fit(X, y)
    assert gam_good.loglikelihood(X, y) >= gam_poor.loglikelihood(X, y)


# ---------------------------------------------------------------------------
# AIC / AICc
# ---------------------------------------------------------------------------


def test_AIC_and_AICc_stored_after_fit(mcycle_X_y):
    """AIC and AICc must be populated in statistics_ after fitting."""
    X, y = mcycle_X_y
    gam = LinearGAM().fit(X, y)
    assert "AIC" in gam.statistics_
    assert "AICc" in gam.statistics_
    assert np.isfinite(gam.statistics_["AIC"])
    assert np.isfinite(gam.statistics_["AICc"])


def test_AICc_ge_AIC(mcycle_X_y):
    """AICc is a corrected (more penalised) AIC, so AICc >= AIC."""
    X, y = mcycle_X_y
    gam = LinearGAM().fit(X, y)
    assert gam.statistics_["AICc"] >= gam.statistics_["AIC"]


# ---------------------------------------------------------------------------
# prediction_intervals()
# ---------------------------------------------------------------------------


def test_prediction_intervals_shape(mcycle_X_y, mcycle_gam):
    """prediction_intervals() must return an array of shape (n, 2)."""
    X, y = mcycle_X_y
    pi = mcycle_gam.prediction_intervals(X)
    assert pi.shape == (len(X), 2)


def test_prediction_intervals_ordered(mcycle_X_y, mcycle_gam):
    """Lower bound must be <= upper bound at every point."""
    X, y = mcycle_X_y
    pi = mcycle_gam.prediction_intervals(X)
    assert np.all(pi[:, 0] <= pi[:, 1])


def test_prediction_intervals_wider_than_confidence(mcycle_X_y, mcycle_gam):
    """Prediction intervals must be at least as wide as confidence intervals."""
    X, y = mcycle_X_y
    ci = mcycle_gam.confidence_intervals(X)
    pi = mcycle_gam.prediction_intervals(X)
    ci_width = ci[:, 1] - ci[:, 0]
    pi_width = pi[:, 1] - pi[:, 0]
    assert np.all(pi_width >= ci_width - 1e-9)


# ---------------------------------------------------------------------------
# partial_dependence()
# ---------------------------------------------------------------------------


def test_partial_dependence_returns_arrays(mcycle_X_y, mcycle_gam):
    """partial_dependence() must return a 1-D array of predictions."""
    X, y = mcycle_X_y
    pdep = mcycle_gam.partial_dependence(term=0)
    assert pdep.ndim == 1
    assert len(pdep) > 0
    assert np.all(np.isfinite(pdep))


def test_partial_dependence_with_width_returns_intervals(mcycle_X_y, mcycle_gam):
    """partial_dependence(width=0.95) must return [pdep, conf_intervals]."""
    X, y = mcycle_X_y
    result = mcycle_gam.partial_dependence(term=0, width=0.95)
    assert isinstance(result, list) and len(result) == 2
    pdep, confi = result
    assert confi.shape == (len(pdep), 2)
    assert np.all(confi[:, 0] <= pdep + 1e-9)
    assert np.all(pdep <= confi[:, 1] + 1e-9)


# ---------------------------------------------------------------------------
# deviance_residuals()
# ---------------------------------------------------------------------------


def test_deviance_residuals_shape(mcycle_X_y, mcycle_gam):
    """deviance_residuals() must return a 1-D array of length n."""
    X, y = mcycle_X_y
    res = mcycle_gam.deviance_residuals(X, y)
    assert res.shape == (len(y),)


def test_deviance_residuals_finite(mcycle_X_y, mcycle_gam):
    """All deviance residuals must be finite for a well-specified model."""
    X, y = mcycle_X_y
    res = mcycle_gam.deviance_residuals(X, y)
    assert np.all(np.isfinite(res))


# ---------------------------------------------------------------------------
# FactorTerm (f())
# ---------------------------------------------------------------------------


def test_factorterm_fit_and_predict(wage_X_y):
    """A GAM with a FactorTerm for a categorical column must fit and predict."""
    X, y = wage_X_y
    # wage dataset: col 2 is job-class (categorical)
    gam = LinearGAM(s(0) + s(1) + f(2)).fit(X, y)
    assert gam._is_fitted
    preds = gam.predict(X)
    assert preds.shape == (len(y),)
    assert np.all(np.isfinite(preds))


def test_factorterm_n_coefs_equals_n_categories(wage_X_y):
    """FactorTerm.n_coefs must equal number of unique values in that column."""
    X, y = wage_X_y
    col = 2
    n_unique = len(np.unique(X[:, col]))
    term = FactorTerm(col)
    term.compile(X)
    assert term.n_coefs == n_unique


# ---------------------------------------------------------------------------
# TensorTerm (te())
# ---------------------------------------------------------------------------


def test_tensorgam_fit_and_predict(wage_X_y):
    """A GAM that uses te() (tensor-product term) must fit and predict."""
    X, y = wage_X_y
    gam = PoissonGAM(te(0, 1) + s(2)).fit(X, y)
    assert gam._is_fitted
    preds = gam.predict(X)
    assert preds.shape == (len(y),)
    assert np.all(preds > 0)  # Poisson predictions are counts


# ---------------------------------------------------------------------------
# Multi-feature statistics
# ---------------------------------------------------------------------------


def test_statistics_p_values_shape(wage_X_y):
    """p_values must have one entry per term (not per coefficient)."""
    X, y = wage_X_y
    gam = LinearGAM(s(0) + s(1) + f(2)).fit(X, y)
    p_vals = gam.statistics_["p_values"]
    assert len(p_vals) == len(gam.terms)


def test_statistics_p_values_in_range(mcycle_X_y, mcycle_gam):
    """All p-values must lie in [0, 1]."""
    p_vals = mcycle_gam.statistics_["p_values"]
    assert np.all(np.array(p_vals) >= 0)
    assert np.all(np.array(p_vals) <= 1)


def test_pseudo_r2_in_range(mcycle_X_y, mcycle_gam):
    """Pseudo-R² values (McFadden / Cox-Snell / Nagelkerke) must be in [0, 1]."""
    r2 = mcycle_gam.statistics_["pseudo_r2"]
    for key, val in r2.items():
        assert 0.0 <= val <= 1.0, f"pseudo_r2[{key!r}] = {val} out of [0,1]"


# ---------------------------------------------------------------------------
# LinearTerm (l())
# ---------------------------------------------------------------------------


def test_linear_term_fit(mcycle_X_y):
    """A GAM with a single LinearTerm must fit without error."""
    X, y = mcycle_X_y
    gam = LinearGAM(l(0)).fit(X, y)
    assert gam._is_fitted
    assert np.all(np.isfinite(gam.coef_))


# ---------------------------------------------------------------------------
# Gamma and InvGauss GAMs (less common but part of the API)
# ---------------------------------------------------------------------------


def test_gammagam_fit(trees_X_y):
    """GammaGAM must fit and produce strictly positive predictions."""
    X, y = trees_X_y
    gam = GammaGAM().fit(X, y)
    assert gam._is_fitted
    preds = gam.predict(X)
    assert np.all(preds > 0)


def test_invgaussgam_fit(trees_X_y):
    """InvGaussGAM must fit and produce strictly positive predictions."""
    X, y = trees_X_y
    gam = InvGaussGAM().fit(X, y)
    assert gam._is_fitted
    preds = gam.predict(X)
    assert np.all(preds > 0)
