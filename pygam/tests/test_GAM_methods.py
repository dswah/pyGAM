# -*- coding: utf-8 -*-

import sys

import numpy as np
import pytest
import scipy as sp

from pygam import *
from pygam.utils import generate_X_grid


@pytest.fixture
def mcycle_gam(mcycle):
    X, y = mcycle
    gam = LinearGAM().fit(X,y)
    return gam

def test_LinearGAM_pdeps_shape(wage):
    """
    check that we get the expected number of partial dependence functions
    """
    X, y = wage
    gam = LinearGAM().fit(X, y)
    pdeps = gam.partial_dependence(X)
    assert(X.shape == pdeps.shape)

def test_LinearGAM_prediction(mcycle, mcycle_gam):
    """
    check that we the predictions we get are correct shape
    """
    X, y = mcycle
    preds = mcycle_gam.predict(X)
    assert(preds.shape == y.shape)

def test_LogisticGAM_accuracy(default):
    """
    check that we can compute accuracy correctly
    """
    X, y = default
    gam = LogisticGAM().fit(X, y)

    preds = gam.predict(X)
    acc0 = (preds == y).mean()
    acc1 = gam.accuracy(X, y)
    assert(acc0 == acc1)

def test_PoissonGAM_exposure(coal):
    """
    check that we can fit a Poisson GAM with exposure, and it scales predictions
    """
    X, y = coal
    gam = PoissonGAM().fit(X, y, exposure=np.ones_like(y))
    assert((gam.predict(X, exposure=np.ones_like(y)*2) == 2 *gam.predict(X)).all())

def test_PoissonGAM_loglike(coal):
    """
    check that our loglikelihood is scaled by exposure

    predictions that are twice as large with twice the exposure
    should have lower loglikelihood
    """
    X, y = coal
    exposure = np.ones_like(y)
    gam_high_var = PoissonGAM().fit(X, y * 2, exposure=exposure * 2)
    gam_low_var = PoissonGAM().fit(X, y, exposure=exposure)

    assert gam_high_var.loglikelihood(X, y * 2, exposure * 2) < gam_low_var.loglikelihood(X, y, exposure)

def test_large_GAM(coal):
    """
    check that we can fit a GAM in py3 when we have more than 50,000 samples
    """
    X = np.linspace(0, 100, 100000)
    y = X**2
    gam = LinearGAM().fit(X, y)
    assert(gam._is_fitted)

def test_summary(mcycle, mcycle_gam):
    """
    check that we can get a summary if we've fitted the model, else not
    """
    X, y = mcycle
    gam = LinearGAM()

    try:
      gam.summary()
    except AttributeError:
      assert(True)

    mcycle_gam.summary()
    assert(True)

def test_more_splines_than_samples(mcycle):
    """
    check that gridsearch returns the expected number of models
    """
    X, y = mcycle
    n = len(X)

    gam = LinearGAM(n_splines=n+1).fit(X, y)
    assert(gam._is_fitted)

def test_deviance_residuals(mcycle, mcycle_gam):
    """
    for linear GAMs, the deviance residuals should be equal to the y - y_pred
    """
    X, y = mcycle
    res = mcycle_gam.deviance_residuals(X, y)
    err = y - mcycle_gam.predict(X)
    assert((res == err).all())

def test_conf_intervals_return_array(mcycle, mcycle_gam):
    """
    make sure that the confidence_intervals method returns an array
    """
    X, y = mcycle
    conf_ints = mcycle_gam.confidence_intervals(X)
    assert(conf_ints.ndim == 2)

def test_conf_intervals_quantiles_width_interchangable(mcycle, mcycle_gam):
    """
    getting confidence_intervals via width or specifying quantiles
    should return the same result
    """
    X, y = mcycle
    conf_ints_a = mcycle_gam.confidence_intervals(X, width=.9)
    conf_ints_b = mcycle_gam.confidence_intervals(X, quantiles=[.05, .95])
    assert(np.allclose(conf_ints_a, conf_ints_b))

def test_conf_intervals_ordered(mcycle, mcycle_gam):
    """
    comfidence intervals returned via width should be ordered
    """
    X, y = mcycle
    conf_ints = mcycle_gam.confidence_intervals(X)
    assert((conf_ints[:,0] <= conf_ints[:,1]).all())

def test_partial_dependence_on_univar_data(mcycle, mcycle_gam):
    """
    partial dependence with univariate data should equal the overall model
    if fit intercept is false
    """
    X, y = mcycle
    gam = LinearGAM(fit_intercept=False).fit(X,y)
    pred = gam.predict(X)
    pdep = gam.partial_dependence(X)
    assert((pred == pdep.ravel()).all())

def test_partial_dependence_on_univar_data2(mcycle, mcycle_gam):
    """
    partial dependence with univariate data should NOT equal the overall model
    if fit intercept is false
    """
    X, y = mcycle
    gam = LinearGAM(fit_intercept=True).fit(X,y)
    pred = gam.predict(X)
    pdep = gam.partial_dependence(X)
    assert((pred != pdep.ravel()).all())

def test_partial_dependence_feature_doesnt_exist(mcycle, mcycle_gam):
    """
    partial dependence should raise ValueError when requesting a nonexistent
    feature
    """
    X, y = mcycle
    try:
        mcycle_gam.partial_dependence(X, feature=10)
    except ValueError:
        assert(True)

def test_summary_returns_12_lines(mcycle_gam):
    """
    check that the summary method works and returns 16 lines like:

    Model Statistics
    -------------------------
    edof               12.321
    AIC              1221.082
    AICc             1224.297
    GCV               611.627
    loglikelihood     -597.22
    deviance          120.679
    scale             510.561

    Pseudo-R^2
    --------------------------
    explained_deviance     0.8
    McFadden             0.288
    McFadden_adj         0.273

    """
    if sys.version_info.major == 2:
        from StringIO import StringIO
    if sys.version_info.major == 3:
        from io import StringIO
    stdout = sys.stdout  #keep a handle on the real standard output
    sys.stdout = StringIO() #Choose a file-like object to write to
    mcycle_gam.summary()
    assert(len(sys.stdout.getvalue().split('\n')) == 16)

def test_is_fitted_predict(mcycle):
    """
    test predict requires fitted model
    """
    X, y = mcycle
    gam = LinearGAM()
    try:
        gam.predict(X)
    except AttributeError:
        assert(True)

def test_is_fitted_predict_mu(mcycle):
    """
    test predict_mu requires fitted model
    """
    X, y = mcycle
    gam = LinearGAM()
    try:
        gam.predict_mu(X)
    except AttributeError:
        assert(True)

def test_is_fitted_dev_resid(mcycle):
    """
    test deviance_residuals requires fitted model
    """
    X, y = mcycle
    gam = LinearGAM()
    try:
        gam.deviance_residuals(X, y)
    except AttributeError:
        assert(True)

def test_is_fitted_conf_intervals(mcycle):
    """
    test confidence_intervals requires fitted model
    """
    X, y = mcycle
    gam = LinearGAM()
    try:
        gam.confidence_intervals(X)
    except AttributeError:
        assert(True)


def test_is_fitted_pdep(mcycle):
    """
    test partial_dependence requires fitted model
    """
    X, y = mcycle
    gam = LinearGAM()
    try:
        gam.partial_dependence(X)
    except AttributeError:
        assert(True)

def test_is_fitted_summary(mcycle):
    """
    test summary requires fitted model
    """
    X, y = mcycle
    gam = LinearGAM()
    try:
        gam.summary()
    except AttributeError:
        assert(True)

def test_set_params_with_external_param():
    """
    test set_params sets a real parameter
    """
    gam = GAM(lam=1)
    gam.set_params(lam=420)
    assert(gam.lam == 420)

def test_set_params_with_hidden_param():
    """
    test set_params should not set any params that are not exposed to the user
    """
    gam = GAM()
    gam.set_params(_lam=420)
    assert(gam._lam != 420)

def test_set_params_with_phony_param():
    """
    test set_params should not set any phony param
    """
    gam = GAM()
    gam.set_params(cat=420)
    assert(not hasattr(gam, 'cat'))

def test_set_params_with_hidden_param_deep():
    """
    test set_params can set hidden params if we use the deep=True
    """
    gam = GAM()
    assert(gam._lam != 420)

    gam.set_params(_lam=420, deep=True)
    assert(gam._lam == 420)

def test_set_params_with_phony_param_force():
    """
    test set_params can set phony params if we use the force=True
    """
    gam = GAM()
    assert(not hasattr(gam, 'cat'))

    gam.set_params(cat=420, force=True)
    assert(gam.cat == 420)

def test_get_params():
    """
    test gam gets our params
    """
    gam = GAM(lam=420)
    params = gam.get_params()
    assert(params['lam'] == 420)

def test_get_params_hidden():
    """
    test gam gets our params only if we do deep=True
    """
    gam = GAM()
    params = gam.get_params()
    assert('_lam' not in list(params.keys()))

    params = gam.get_params(deep=True)
    assert('_lam' in list(params.keys()))


class TestSamplingFromPosterior(object):

    def test_drawing_samples_from_unfitted_model(self, mcycle, mcycle_gam):
        X, y = mcycle
        gam = LinearGAM()

        with pytest.raises(AttributeError):
            gam.sample(X, y)

        with pytest.raises(AttributeError):
            gam._sample_coef(X, y)

        with pytest.raises(AttributeError):
            gam._bootstrap_samples_of_smoothing(X, y)

        assert mcycle_gam._is_fitted

        mcycle_gam.sample(X, y, n_draws=2)
        mcycle_gam._sample_coef(X, y, n_draws=2)
        mcycle_gam._bootstrap_samples_of_smoothing(X, y, n_bootstraps=1)
        assert True

    def test_sample_quantity(self, mcycle, mcycle_gam):
        X, y = mcycle
        for quantity in ['coefficients', 'response']:
            with pytest.raises(ValueError):
                mcycle_gam.sample(X, y, quantity=quantity, n_draws=2)
        for quantity in ['coef', 'mu', 'y']:
            mcycle_gam.sample(X, y, quantity=quantity, n_draws=2)
            assert True

    def test_shape_of_random_samples(self, mcycle, mcycle_gam):
        X, y = mcycle
        n_samples = len(X)
        n_draws = 5

        sample_coef = mcycle_gam.sample(X, y, quantity='coef', n_draws=n_draws)
        sample_mu = mcycle_gam.sample(X, y, quantity='mu', n_draws=n_draws)
        sample_y = mcycle_gam.sample(X, y, quantity='y', n_draws=n_draws)
        assert sample_coef.shape == (n_draws, len(mcycle_gam.coef_))
        assert sample_mu.shape == (n_draws, n_samples)
        assert sample_y.shape == (n_draws, n_samples)

        XX = generate_X_grid(mcycle_gam)
        n_samples_in_grid = len(XX)
        sample_coef = mcycle_gam.sample(X, y, quantity='coef', n_draws=n_draws,
                                        sample_at_X=XX)
        sample_mu = mcycle_gam.sample(X, y, quantity='mu', n_draws=n_draws,
                                        sample_at_X=XX)
        sample_y = mcycle_gam.sample(X, y, quantity='y', n_draws=n_draws,
                                        sample_at_X=XX)

        assert sample_coef.shape == (n_draws, len(mcycle_gam.coef_))
        assert sample_mu.shape == (n_draws, n_samples_in_grid)
        assert sample_y.shape == (n_draws, n_samples_in_grid)

    def test_shape_bootstrap_samples_of_smoothing(self, mcycle, mcycle_gam):
        X, y = mcycle

        for n_bootstraps in [1, 2]:
            coef_bootstraps, cov_bootstraps = (
                mcycle_gam._bootstrap_samples_of_smoothing(
                    X, y, n_bootstraps=n_bootstraps))
            assert len(coef_bootstraps) == len(cov_bootstraps) == n_bootstraps
            for coef, cov in zip(coef_bootstraps, cov_bootstraps):
                assert coef.shape == mcycle_gam.coef_.shape
                assert cov.shape == mcycle_gam.statistics_['cov'].shape

            for n_draws in [1, 2]:
                coef_draws = mcycle_gam._simulate_coef_from_bootstraps(
                    n_draws, coef_bootstraps, cov_bootstraps)
                assert coef_draws.shape == (n_draws, len(mcycle_gam.coef_))

    def test_bad_sample_params(self, mcycle, mcycle_gam):
        X, y = mcycle
        with pytest.raises(ValueError):
            mcycle_gam.sample(X, y, n_draws=0)
        with pytest.raises(ValueError):
            mcycle_gam.sample(X, y, n_bootstraps=0)


def test_prediction_interval_unknown_scale():
    """
    the prediction intervals should be correct to a few decimal places
    we test at a large sample limit, where the t distribution becomes normal
    """
    n = 1000000
    X = np.linspace(0,1,n)
    y = np.random.randn(n)

    gam_a = LinearGAM(fit_linear=True, fit_splines=False).fit(X, y)
    gam_b = LinearGAM(n_splines=4).fit(X, y)

    XX = generate_X_grid(gam_a)
    intervals_a = gam_a.prediction_intervals(XX, quantiles=[0.1, .9]).mean(axis=0)
    intervals_b = gam_b.prediction_intervals(XX, quantiles=[0.1, .9]).mean(axis=0)

    assert np.allclose(intervals_a[0], sp.stats.norm.ppf(0.1), atol=0.01)
    assert np.allclose(intervals_a[1], sp.stats.norm.ppf(0.9), atol=0.01)

    assert np.allclose(intervals_b[0], sp.stats.norm.ppf(0.1), atol=0.01)
    assert np.allclose(intervals_b[1], sp.stats.norm.ppf(0.9), atol=0.01)

def test_prediction_interval_known_scale():
    """
    the prediction intervals should be correct to a few decimal places
    we test at a large sample limit.
    """
    n = 1000000
    X = np.linspace(0,1,n)
    y = np.random.randn(n)

    gam_a = LinearGAM(fit_linear=True, fit_splines=False, scale=1.).fit(X, y)
    gam_b = LinearGAM(n_splines=4, scale=1.).fit(X, y)

    XX = generate_X_grid(gam_a)
    intervals_a = gam_a.prediction_intervals(XX, quantiles=[0.1, .9]).mean(axis=0)
    intervals_b = gam_b.prediction_intervals(XX, quantiles=[0.1, .9]).mean(axis=0)

    assert np.allclose(intervals_a[0], sp.stats.norm.ppf(0.1), atol=0.01)
    assert np.allclose(intervals_a[1], sp.stats.norm.ppf(0.9), atol=0.01)

    assert np.allclose(intervals_b[0], sp.stats.norm.ppf(0.1), atol=0.01)
    assert np.allclose(intervals_b[1], sp.stats.norm.ppf(0.9), atol=0.01)
