from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from pygam import LinearGAM, LogisticGAM, f, s
from pygam.utils import (
    check_iterable_depth,
    check_X,
    check_X_y,
    check_y,
    sig_code,
    cholesky,
    make_2d,
    check_array,
    check_lengths,
    check_param,
    round_to_n_decimal_places,
    space_row,
    gen_edge_knots,
)

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
        with patch(
            "pygam.utils.SKSPIMPORT", new=False
        ) as SKSPIMPORT_patch:  # noqa: E501, F841
            from pygam.utils import SKSPIMPORT

            assert SKSPIMPORT is False

            X, y = mcycle_X_y
            assert LinearGAM().fit(X, y)._is_fitted


def test_make_2d_warning():
    with pytest.warns(UserWarning):
        make_2d([1, 2, 3], verbose=True)


def test_check_array_wrong_n_feats():
    with pytest.raises(ValueError, match="must have 2 features"):
        check_array(np.ones((5, 3)), n_feats=2)


def test_check_lengths_mismatch():
    with pytest.raises(ValueError, match="Inconsistent data lengths"):
        check_lengths([1, 2], [1])


def test_check_param_invalid_value():
    with pytest.raises(ValueError):
        check_param("not_a_number", "param", "float")


def test_check_param_depth_exceeded():
    with pytest.raises(TypeError):
        check_param([[[1]]], "param", "float", max_depth=2)


def test_check_param_not_iterable():
    with pytest.raises(TypeError):
        check_param([1], "param", "float", iterable=False)


def test_check_param_wrong_dtype():
    with pytest.raises(ValueError):
        check_param(["a", "b"], "param", "float")


def test_check_param_no_constraint_msg():
    with pytest.raises(ValueError) as exc:
        check_param("a", "param", "float", constraint=None)
    assert "could not convert string to float" in str(exc.value)


def test_round_to_n_decimal_places_scientific():
    # should return as is
    val = float("1e-10")
    assert round_to_n_decimal_places(val) == val


def test_space_row_negative_width():
    assert space_row("a", "b", total_width=-3) == "a   b"


def test_gen_edge_knots_unsupported():
    with pytest.raises(ValueError, match="unsupported dtype"):
        gen_edge_knots([1, 2], "invalid")


def test_gen_edge_knots_constant():
    with pytest.warns(UserWarning):
        gen_edge_knots([1, 1], "numerical", verbose=True)


def test_cholesky_sparse_sklearn():
    import scipy as sp

    A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    L = cholesky(A, sparse=True, verbose=False)
    assert sp.sparse.issparse(L)


def test_cholesky_not_pos_def():
    A = np.array([[0, 0], [0, 0]])
    from pygam.utils import NotPositiveDefiniteError

    with pytest.raises(NotPositiveDefiniteError):
        cholesky(A, sparse=False, verbose=False)


def test_cholesky_skspimport_true():
    import numpy as np
    import scipy as sp

    with patch("pygam.utils.SKSPIMPORT", new=True):

        class MockF:
            def P(self):
                return np.array([0, 1])

            def L(self):
                return sp.sparse.csc_array(np.array([[1, 0], [0, 1]]))

        with patch("pygam.utils.spcholesky", return_value=MockF(), create=True):
            from pygam.utils import cholesky

            A = np.array([[2, -1], [-1, 2]])
            L = cholesky(A, sparse=True, verbose=False)
            assert sp.sparse.issparse(L)
            L_dense = cholesky(A, sparse=False, verbose=False)
            assert not sp.sparse.issparse(L_dense)


def test_b_spline_basis_dense():
    import numpy as np
    import scipy as sp
    from pygam.utils import b_spline_basis
    x = np.linspace(0, 1, 10)
    basis = b_spline_basis(x, edge_knots=[0, 1], sparse=False)
    assert not sp.sparse.issparse(basis)

def test_combine():
    from pygam.utils import combine
    res = combine([[1, 2], [3]])
    assert len(res) == 2

def test_b_spline_basis_errors():
    from pygam.utils import b_spline_basis
    import numpy as np
    import pytest
    with pytest.raises(ValueError, match="n_splines must be int >= 1"):
        b_spline_basis([1], [0, 1], n_splines=0)
    with pytest.raises(ValueError, match="spline_order must be int >= 1"):
        b_spline_basis([1], [0, 1], n_splines=3, spline_order=-1)
    with pytest.raises(ValueError, match="n_splines must be >="):
        b_spline_basis([1], [0, 1], n_splines=2, spline_order=2)

def test_tensor_product_mismatch():
    from pygam.utils import tensor_product
    import numpy as np
    import pytest
    with pytest.raises(ValueError, match="same number of samples"):
        tensor_product(np.ones((2, 2)), np.ones((3, 2)))

def test_tensor_product_reshape_false():
    from pygam.utils import tensor_product
    import numpy as np
    res = tensor_product(np.ones((2, 2)), np.ones((2, 3)), reshape=False)
    assert res.shape == (2, 2, 3)
