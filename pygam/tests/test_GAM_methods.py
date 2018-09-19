# -*- coding: utf-8 -*-

import sys

import numpy as np
import pytest
import scipy as sp

from pygam import *


def test_LinearGAM_prediction(mcycle_X_y, mcycle_gam):
    """
    check that we the predictions we get are correct shape
    """
    X, y = mcycle_X_y
    preds = mcycle_gam.predict(X)
    assert(preds.shape == y.shape)

def test_LogisticGAM_accuracy(default_X_y):
    """
    check that we can compute accuracy correctly
    """
    X, y = default_X_y
    gam = LogisticGAM().fit(X, y)

    preds = gam.predict(X)
    acc0 = (preds == y).mean()
    acc1 = gam.accuracy(X, y)
    assert(acc0 == acc1)

def test_PoissonGAM_exposure(coal_X_y):
    """
    check that we can fit a Poisson GAM with exposure, and it scales predictions
    """
    X, y = coal_X_y
    gam = PoissonGAM().fit(X, y, exposure=np.ones_like(y))
    assert((gam.predict(X, exposure=np.ones_like(y)*2) == 2 *gam.predict(X)).all())

def test_PoissonGAM_loglike(coal_X_y):
    """
    check that our loglikelihood is scaled by exposure

    predictions that are twice as large with twice the exposure
    should have lower loglikelihood
    """
    X, y = coal_X_y
    exposure = np.ones_like(y)
    gam_high_var = PoissonGAM().fit(X, y * 2, exposure=exposure * 2)
    gam_low_var = PoissonGAM().fit(X, y, exposure=exposure)

    assert gam_high_var.loglikelihood(X, y * 2, exposure * 2) < gam_low_var.loglikelihood(X, y, exposure)

def test_large_GAM(coal_X_y):
    """
    check that we can fit a GAM in py3 when we have more than 50,000 samples
    """
    X = np.linspace(0, 100, 100000)
    y = X**2
    gam = LinearGAM().fit(X, y)
    assert(gam._is_fitted)

def test_summary(mcycle_X_y, mcycle_gam):
    """
    check that we can get a summary if we've fitted the model, else not
    """
    X, y = mcycle_X_y
    gam = LinearGAM()

    try:
      gam.summary()
    except AttributeError:
      assert(True)

    mcycle_gam.summary()
    assert(True)

def test_more_splines_than_samples(mcycle_X_y):
    """
    check that gridsearch returns the expected number of models
    """
    X, y = mcycle_X_y
    n = len(X)

    gam = LinearGAM(s(0, n_splines=n+1)).fit(X, y)
    assert(gam._is_fitted)

    # TODO here is our bug:
    # we cannot display the term-by-term effective DoF because we have fewer
    # values than coefficients
    assert len(gam.statistics_['edof_per_coef']) < len(gam.coef_)
    gam.summary()

def test_deviance_residuals(mcycle_X_y, mcycle_gam):
    """
    for linear GAMs, the deviance residuals should be equal to the y - y_pred
    """
    X, y = mcycle_X_y
    res = mcycle_gam.deviance_residuals(X, y)
    err = y - mcycle_gam.predict(X)
    assert((res == err).all())

def test_conf_intervals_return_array(mcycle_X_y, mcycle_gam):
    """
    make sure that the confidence_intervals method returns an array
    """
    X, y = mcycle_X_y
    conf_ints = mcycle_gam.confidence_intervals(X)
    assert(conf_ints.ndim == 2)

def test_conf_intervals_quantiles_width_interchangable(mcycle_X_y, mcycle_gam):
    """
    getting confidence_intervals via width or specifying quantiles
    should return the same result
    """
    X, y = mcycle_X_y
    conf_ints_a = mcycle_gam.confidence_intervals(X, width=.9)
    conf_ints_b = mcycle_gam.confidence_intervals(X, quantiles=[.05, .95])
    assert(np.allclose(conf_ints_a, conf_ints_b))

def test_conf_intervals_ordered(mcycle_X_y, mcycle_gam):
    """
    comfidence intervals returned via width should be ordered
    """
    X, y = mcycle_X_y
    conf_ints = mcycle_gam.confidence_intervals(X)
    assert((conf_ints[:,0] <= conf_ints[:,1]).all())

def test_summary_returns_12_lines(mcycle_gam):
    """
    check that the summary method works and returns 24 lines like:

    LinearGAM
    =============================================== ==========================================================
    Distribution:                        NormalDist Effective DoF:                                     11.2495
    Link Function:                     IdentityLink Log Likelihood:                                   -952.605
    Number of Samples:                          133 AIC:                                             1929.7091
                                                    AICc:                                            1932.4197
                                                    GCV:                                              605.6546
                                                    Scale:                                            514.2013
                                                    Pseudo R-Squared:                                   0.7969
    ==========================================================================================================
    Feature Function   Data Type      Num Splines   Spline Order  Linear Fit  Lambda     P > x      Sig. Code
    ================== ============== ============= ============= =========== ========== ========== ==========
    feature 1          numerical      25            3             False       1.0        3.43e-03   **
    intercept                                                                            6.85e-02   .
    ==========================================================================================================
    Significance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    WARNING: Fitting splines and a linear function to a feature introduces a model identifiability problem
             which can cause p-values to appear significant when they are not.

    WARNING: p-values calculated in this manner behave correctly for un-penalized models or models with
             known smoothing parameters, but when smoothing parameters have been estimated, the p-values
             are typically lower than they should be, meaning that the tests reject the null too readily.
    """
    if sys.version_info.major == 2:
        from StringIO import StringIO
    if sys.version_info.major == 3:
        from io import StringIO
    stdout = sys.stdout  #keep a handle on the real standard output
    sys.stdout = StringIO() #Choose a file-like object to write to
    mcycle_gam.summary()
    assert(len(sys.stdout.getvalue().split('\n')) == 24)

def test_is_fitted_predict(mcycle_X_y):
    """
    test predict requires fitted model
    """
    X, y = mcycle_X_y
    gam = LinearGAM()
    with pytest.raises(AttributeError):
        gam.predict(X)

def test_is_fitted_predict_mu(mcycle_X_y):
    """
    test predict_mu requires fitted model
    """
    X, y = mcycle_X_y
    gam = LinearGAM()
    with pytest.raises(AttributeError):
        gam.predict_mu(X)

def test_is_fitted_dev_resid(mcycle_X_y):
    """
    test deviance_residuals requires fitted model
    """
    X, y = mcycle_X_y
    gam = LinearGAM()
    with pytest.raises(AttributeError):
        gam.deviance_residuals(X, y)

def test_is_fitted_conf_intervals(mcycle_X_y):
    """
    test confidence_intervals requires fitted model
    """
    X, y = mcycle_X_y
    gam = LinearGAM()
    with pytest.raises(AttributeError):
        gam.confidence_intervals(X)

def test_is_fitted_pdep(mcycle_X_y):
    """
    test partial_dependence requires fitted model
    """
    gam = LinearGAM()
    with pytest.raises(AttributeError):
        gam.partial_dependence(term=0)

def test_is_fitted_summary(mcycle_X_y):
    """
    test summary requires fitted model
    """
    X, y = mcycle_X_y
    gam = LinearGAM()
    with pytest.raises(AttributeError):
        gam.summary()

def test_set_params_with_external_param():
    """
    test set_params sets a real parameter
    """
    gam = GAM(lam=1)
    gam.set_params(lam=420)
    assert(gam.lam == 420)

def test_set_params_with_phony_param():
    """
    test set_params should not set any phony param
    """
    gam = GAM()
    gam.set_params(cat=420)
    assert(not hasattr(gam, 'cat'))

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


class TestSamplingFromPosterior(object):

    def test_drawing_samples_from_unfitted_model(self, mcycle_X_y, mcycle_gam):
        X, y = mcycle_X_y
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

    def test_sample_quantity(self, mcycle_X_y, mcycle_gam):
        X, y = mcycle_X_y
        for quantity in ['coefficients', 'response']:
            with pytest.raises(ValueError):
                mcycle_gam.sample(X, y, quantity=quantity, n_draws=2)
        for quantity in ['coef', 'mu', 'y']:
            mcycle_gam.sample(X, y, quantity=quantity, n_draws=2)
            assert True

    def test_shape_of_random_samples(self, mcycle_X_y, mcycle_gam):
        X, y = mcycle_X_y
        n_samples = len(X)
        n_draws = 5

        sample_coef = mcycle_gam.sample(X, y, quantity='coef', n_draws=n_draws)
        sample_mu = mcycle_gam.sample(X, y, quantity='mu', n_draws=n_draws)
        sample_y = mcycle_gam.sample(X, y, quantity='y', n_draws=n_draws)
        assert sample_coef.shape == (n_draws, len(mcycle_gam.coef_))
        assert sample_mu.shape == (n_draws, n_samples)
        assert sample_y.shape == (n_draws, n_samples)

        n_samples_in_grid = 500
        idxs = np.random.choice(np.arange(len(X)), n_samples_in_grid)
        XX = X[idxs]

        sample_coef = mcycle_gam.sample(X, y, quantity='coef', n_draws=n_draws,
                                        sample_at_X=XX)
        sample_mu = mcycle_gam.sample(X, y, quantity='mu', n_draws=n_draws,
                                        sample_at_X=XX)
        sample_y = mcycle_gam.sample(X, y, quantity='y', n_draws=n_draws,
                                        sample_at_X=XX)

        assert sample_coef.shape == (n_draws, len(mcycle_gam.coef_))
        assert sample_mu.shape == (n_draws, n_samples_in_grid)
        assert sample_y.shape == (n_draws, n_samples_in_grid)

    def test_shape_bootstrap_samples_of_smoothing(self, mcycle_X_y, mcycle_gam):
        X, y = mcycle_X_y

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

    def test_bad_sample_params(self, mcycle_X_y, mcycle_gam):
        X, y = mcycle_X_y
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

    gam_a = LinearGAM(terms=l(0)).fit(X, y)
    gam_b = LinearGAM(s(0, n_splines=4)).fit(X, y)

    XX = gam_a.generate_X_grid(term=0)
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

    gam_a = LinearGAM(terms=l(0), scale=1.).fit(X, y)
    gam_b = LinearGAM(s(0, n_splines=4), scale=1.).fit(X, y)

    XX = gam_a.generate_X_grid(term=0)
    intervals_a = gam_a.prediction_intervals(XX, quantiles=[0.1, .9]).mean(axis=0)
    intervals_b = gam_b.prediction_intervals(XX, quantiles=[0.1, .9]).mean(axis=0)

    assert np.allclose(intervals_a[0], sp.stats.norm.ppf(0.1), atol=0.01)
    assert np.allclose(intervals_a[1], sp.stats.norm.ppf(0.9), atol=0.01)

    assert np.allclose(intervals_b[0], sp.stats.norm.ppf(0.1), atol=0.01)
    assert np.allclose(intervals_b[1], sp.stats.norm.ppf(0.9), atol=0.01)

def test_pvalue_rejects_useless_feature(wage_X_y):
    """
    check that a p-value can reject a useless feature
    """
    X, y = wage_X_y

    # add empty feature
    X = np.c_[X, np.arange(X.shape[0])]
    gam = LinearGAM(s(0) + s(1) + f(2) + s(3)).fit(X, y)

    # now do the test, with some safety
    p_values = gam._estimate_p_values()
    print(p_values)
    assert(p_values[-2] > .5) # because -1 is intercept

def test_fit_quantile_is_close_enough(head_circumference_X_y):
    """see that we get close to the desired quantile

    and check that repeating on an already fitted returns the same
    """
    X, y = head_circumference_X_y

    quantile = 0.99
    tol = 1e-4

    gam = ExpectileGAM().fit_quantile(X, y, quantile=quantile, max_iter=20, tol=tol)
    ratio = gam._get_quantile_ratio(X, y)

    assert np.abs(ratio - quantile) <= tol

    # now check if we had to refit
    gam2 = gam.fit_quantile(X, y, quantile=quantile, max_iter=20, tol=tol)

    assert gam == gam2


def test_fit_quantile_NOT_close_enough(head_circumference_X_y):
    """see that we DO NOT get close to the desired quantile
    """
    X, y = head_circumference_X_y

    quantile = 0.99
    tol = 1e-5

    gam = ExpectileGAM().fit_quantile(X, y, quantile=quantile, max_iter=1, tol=tol)
    ratio = gam._get_quantile_ratio(X, y)

    assert np.abs(ratio - quantile) > tol

def test_fit_quantile_raises_ValueError(head_circumference_X_y):
    """see that we DO NOT get fit on bad argument requests
    """
    X, y = head_circumference_X_y

    with pytest.raises(ValueError):
        ExpectileGAM().fit_quantile(X, y, quantile=0)

    with pytest.raises(ValueError):
        ExpectileGAM().fit_quantile(X, y, quantile=1)

    with pytest.raises(ValueError):
        ExpectileGAM().fit_quantile(X, y, quantile=-0.1)

    with pytest.raises(ValueError):
        ExpectileGAM().fit_quantile(X, y, quantile=1.1)

    with pytest.raises(ValueError):
        ExpectileGAM().fit_quantile(X, y, tol=0, quantile=0.5)

    with pytest.raises(ValueError):
        ExpectileGAM().fit_quantile(X, y, tol=-0.1, quantile=0.5)

    with pytest.raises(ValueError):
        ExpectileGAM().fit_quantile(X, y, max_iter=0, quantile=0.5)

    with pytest.raises(ValueError):
        ExpectileGAM().fit_quantile(X, y, max_iter=-1, quantile=0.5)

class TestRegressions(object):
    def test_pvalue_invariant_to_scale(self, wage_X_y):
        """
        regression test.

        a bug made the F-statistic sensitive to scale changes, when it should be invariant.

        check that a p-value should not change when we change the scale of the response
        """
        X, y = wage_X_y

        gamA = LinearGAM(s(0) + s(1) + f(2)).fit(X, y * 1000000)
        gamB = LinearGAM(s(0) + s(1) + f(2)).fit(X, y)

        assert np.allclose(gamA.statistics_['p_values'], gamB.statistics_['p_values'])

    def test_2d_y_still_allow_fitting_in_PoissonGAM(self, coal_X_y):
        """
        regression test.

        there was a bug where we forgot to check the y_array before converting
        exposure to weights.
        """
        X, y = coal_X_y
        two_d_data = np.ones_like(y).ravel()[:, None]

        # 2d y should cause no problems now
        gam = PoissonGAM().fit(X, y[:, None])
        assert gam._is_fitted

        # 2d weghts should cause no problems now
        gam = PoissonGAM().fit(X, y, weights=two_d_data)
        assert gam._is_fitted

        # 2d exposure should cause no problems now
        gam = PoissonGAM().fit(X, y, exposure=two_d_data)
        assert gam._is_fitted

    def test_non_int_exposure_produced_no_inf_in_PoissonGAM_ll(self, coal_X_y):
        """
        regression test.

        there was a bug where we forgot to round the rescaled counts before
        computing the loglikelihood. since Poisson requires integer observations,
        small numerical errors caused the pmf to return -inf, which shows up
        in the loglikelihood computations, AIC, AICc..
        """
        X, y = coal_X_y

        rate = 1.2 + np.cos(np.linspace(0, 2. * np.pi, len(y)))

        gam = PoissonGAM().fit(X, y, exposure=rate)

        assert np.isfinite(gam.statistics_['loglikelihood'])

    def test_initial_estimate_runs_for_int_obseravtions(self, toy_classification_X_y):
        """
        regression test

        ._initial_estimate would fail when trying to add small numbers to
        integer observations

        casting the observations to float in that method fixes that
        """
        X, y = toy_classification_X_y
        gam = LogisticGAM().fit(X, y)
        assert gam._is_fitted

    def test_r_squared_for_new_dataset(self, mcycle_gam, mcycle_X_y):
        """
        regression test

        estimate r squared used to refer to a non-existant method when `mu=None`
        """
        X, y = mcycle_X_y
        mcycle_gam._estimate_r2(X, y)
