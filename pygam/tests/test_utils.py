from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from pygam import LinearGAM, LogisticGAM, f, s
from pygam.utils import check_iterable_depth, check_X, check_X_y, check_y, sig_code

# TODO check dtypes works as expected
# TODO checkX, checky, check XY expand as needed, call out bad domain


@pytest.fixture
def wage_gam(wage_X_y):
    X, y = wage_X_y
    gam = LinearGAM(s(0) + s(1) + f(2)).fit(X, y)
    return gam


@pytest.fixture
def default_gam(default_X_y):
    X, y = default_X_y
    gam = LogisticGAM().fit(X, y)
    return gam


def test_check_X_categorical_prediction_exceeds_training(wage_X_y, wage_gam):
    """
    if our categorical variable is outside the training range
    we should get an error
    """
    X, y = wage_X_y  # last feature is categorical
    gam = wage_gam

    # get edge knots for last feature
    eks = gam.edge_knots_[-1]

    # add 1 to all Xs, thus pushing some X past the max value
    X[:, -1] = eks[-1] + 1

    with pytest.raises(ValueError):
        gam.predict(X)


def test_check_y_not_int_not_float(wage_X_y, wage_gam):
    """y must be int or float, or we should get a value error"""
    X, y = wage_X_y
    y_str = ["hi"] * len(y)

    with pytest.raises(ValueError):
        check_y(y_str, wage_gam.link, wage_gam.distribution)


def test_check_y_casts_to_numerical(wage_X_y, wage_gam):
    """check_y will try to cast data to numerical types"""
    X, y = wage_X_y
    y = y.astype("object")

    y = check_y(y, wage_gam.link, wage_gam.distribution)
    assert y.dtype == "float"


def test_check_y_not_min_samples(wage_X_y, wage_gam):
    """check_y expects a minimum number of samples"""
    X, y = wage_X_y

    with pytest.raises(ValueError):
        check_y(
            y,
            wage_gam.link,
            wage_gam.distribution,
            min_samples=len(y) + 1,
            verbose=False,
        )


def test_check_y_not_in_domain_link(default_X_y, default_gam):
    """if you give labels outide of the links domain, check_y will raise an error"""
    X, y = default_X_y

    with pytest.raises(ValueError):
        check_y(y + 0.1, default_gam.link, default_gam.distribution, verbose=False)


def test_check_X_not_int_not_float():
    """X  must be an in or a float"""
    with pytest.raises(ValueError):
        check_X(["hi"], verbose=False)


def test_check_X_too_many_dims():
    """check_X accepts at most 2D inputs"""
    with pytest.raises(ValueError):
        check_X(np.ones((5, 4, 3)))


def test_check_X_not_min_samples():
    with pytest.raises(ValueError):
        check_X(np.ones(5), min_samples=6, verbose=False)


def test_check_X_y_different_lengths():
    with pytest.raises(ValueError):
        check_X_y(np.ones(5), np.ones(4))


def test_input_data_after_fitting(mcycle_X_y):
    """
    our check_X and check_y functions should be invoked
    any time external data is input to the model
    """
    X, y = mcycle_X_y
    weights = np.ones_like(y)

    X_nan = deepcopy(X)
    X_nan[0] = X_nan[0] * np.nan

    y_nan = deepcopy(y.values)
    y_nan[0] = y_nan[0] * np.nan

    weights_nan = deepcopy(weights)
    weights_nan[0] = weights_nan[0] * np.nan

    gam = LinearGAM()

    with pytest.raises(ValueError):
        gam.fit(X_nan, y, weights)
    with pytest.raises(ValueError):
        gam.fit(X, y_nan, weights)
    with pytest.raises(ValueError):
        gam.fit(X, y, weights_nan)
    gam = gam.fit(X, y)

    # test X is nan
    with pytest.raises(ValueError):
        gam.predict(X_nan)
    with pytest.raises(ValueError):
        gam.predict_mu(X_nan)
    with pytest.raises(ValueError):
        gam.confidence_intervals(X_nan)
    with pytest.raises(ValueError):
        gam.prediction_intervals(X_nan)
    with pytest.raises(ValueError):
        gam.partial_dependence(X_nan)
    with pytest.raises(ValueError):
        gam.deviance_residuals(X_nan, y, weights)
    with pytest.raises(ValueError):
        gam.loglikelihood(X_nan, y, weights)
    with pytest.raises(ValueError):
        gam.gridsearch(X_nan, y, weights)
    with pytest.raises(ValueError):
        gam.sample(X_nan, y)

    # test y is nan
    with pytest.raises(ValueError):
        gam.deviance_residuals(X, y_nan, weights)
    with pytest.raises(ValueError):
        gam.loglikelihood(X, y_nan, weights)
    with pytest.raises(ValueError):
        gam.gridsearch(X, y_nan, weights)
    with pytest.raises(ValueError):
        gam.sample(X, y_nan, weights=weights, n_bootstraps=2)

    # test weights is nan
    with pytest.raises(ValueError):
        gam.deviance_residuals(X, y, weights_nan)
    with pytest.raises(ValueError):
        gam.loglikelihood(X, y, weights_nan)
    with pytest.raises(ValueError):
        gam.gridsearch(X, y, weights_nan)
    with pytest.raises(ValueError):
        gam.sample(X, y, weights=weights_nan, n_bootstraps=2)


def test_catch_chol_pos_def_error(default_X_y):
    """
    regression test

    doing a gridsearch with a poorly conditioned penalty matrix should not crash
    """
    X, y = default_X_y
    LogisticGAM().gridsearch(X, y, lam=np.logspace(10, 12, 3))


def test_pvalue_sig_codes():
    """make sure we get the codes we expect"""
    with pytest.raises(AssertionError):
        sig_code(-1)

    assert sig_code(0) == "***"
    assert sig_code(0.00101) == "**"
    assert sig_code(0.0101) == "*"
    assert sig_code(0.0501) == "."
    assert sig_code(0.101) == " "


def test_b_spline_basis_extrapolates(mcycle_X_y):
    X, y = mcycle_X_y
    gam = LinearGAM().fit(X, y)

    slopes = []

    X = gam.generate_X_grid(term=0, n=50000)
    y = gam.predict(X)
    slopes.append((y[1] - y[0]) / (X[1] - X[0]))

    mean = X.mean()
    X -= mean
    X *= 1.1
    X += mean

    y = gam.predict(X)
    slopes.append((y[1] - y[0]) / (X[1] - X[0]))

    assert np.allclose(slopes[0], slopes[1], atol=1e-4)


def test_iterable_depth():
    it = [[[3]]]
    assert check_iterable_depth(it) == 3
    assert check_iterable_depth(it, max_depth=2) == 2


def test_no_SKSPIMPORT(mcycle_X_y):
    """make sure our module work with and without scikit-sparse"""
    from pygam.utils import SKSPIMPORT

    if SKSPIMPORT:
        with patch("pygam.utils.SKSPIMPORT", new=False) as SKSPIMPORT_patch:  # noqa: E501, F841
            from pygam.utils import SKSPIMPORT

            assert SKSPIMPORT is False

            X, y = mcycle_X_y
            assert LinearGAM().fit(X, y)._is_fitted
