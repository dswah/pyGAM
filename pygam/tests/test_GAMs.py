import pytest

from pygam import (
    GAM,
    ExpectileGAM,
    GammaGAM,
    InvGaussGAM,
    LinearGAM,
    LogisticGAM,
    PoissonGAM,
)


def test_can_build_sub_models():
    """
    check that the inits of all the sub-models are correct
    """
    LinearGAM()
    LogisticGAM()
    PoissonGAM()
    GammaGAM()
    InvGaussGAM()
    ExpectileGAM()
    assert True


def test_LinearGAM_uni(mcycle_X_y):
    """
    check that we can fit a Linear GAM on real, univariate data
    """
    X, y = mcycle_X_y
    gam = LinearGAM().fit(X, y)
    assert gam._is_fitted


def test_LinearGAM_multi(wage_X_y):
    """
    check that we can fit a Linear GAM on real, multivariate data
    """
    X, y = wage_X_y
    gam = LinearGAM().fit(X, y)
    assert gam._is_fitted


def test_LogisticGAM(default_X_y):
    """
    check that we can fit a Logistic GAM on real data
    """
    X, y = default_X_y
    gam = LogisticGAM().fit(X, y)
    assert gam._is_fitted


def test_PoissonGAM(coal_X_y):
    """
    check that we can fit a Poisson GAM on real data
    """
    X, y = coal_X_y
    gam = PoissonGAM().fit(X, y)
    assert gam._is_fitted


def test_InvGaussGAM(trees_X_y):
    """
    check that we can fit a InvGauss GAM on real data
    """
    X, y = trees_X_y
    gam = InvGaussGAM().fit(X, y)
    assert gam._is_fitted


def test_GammaGAM(trees_X_y):
    """
    check that we can fit a Gamma GAM on real data
    """
    X, y = trees_X_y
    gam = GammaGAM().fit(X, y)
    assert gam._is_fitted


def test_CustomGAM(trees_X_y):
    """
    check that we can fit a Custom GAM on real data
    """
    X, y = trees_X_y
    gam = GAM(distribution="gamma", link="inverse").fit(X, y)
    assert gam._is_fitted


def test_ExpectileGAM_uni(mcycle_X_y):
    """
    check that we can fit an Expectile GAM on real, univariate data
    """
    X, y = mcycle_X_y
    gam = ExpectileGAM().fit(X, y)
    assert gam._is_fitted


def test_ExpectileGAM_bad_expectiles(mcycle_X_y):
    """
    check that get errors for unacceptable expectiles
    """
    X, y = mcycle_X_y
    with pytest.raises(ValueError):
        ExpectileGAM(expectile=0).fit(X, y)
    with pytest.raises(ValueError):
        ExpectileGAM(expectile=1).fit(X, y)
    with pytest.raises(ValueError):
        ExpectileGAM(expectile=-0.1).fit(X, y)
    with pytest.raises(ValueError):
        ExpectileGAM(expectile=1.1).fit(X, y)


def test_normal_dist_log_pdf_weights():
    """
    Regression test for Issue #457: NormalDist.log_pdf must apply
    weights to variance, not linearly to standard deviation.
    """
    import numpy as np
    import scipy.stats as st

    from pygam.distributions import NormalDist

    scale = 1.0
    weights = np.array([4.0])
    y = np.array([0.0])
    mu = np.array([0.0])

    dist = NormalDist(scale=scale)
    actual = dist.log_pdf(y, mu, weights=weights)[0]

    # Standard deviation used by scipy should be scale / sqrt(w)
    # For scale=1, w=4, the SD is 0.5
    expected_sd = scale / np.sqrt(weights)
    expected = st.norm.logpdf(y, loc=mu, scale=expected_sd)[0]

    assert np.isclose(actual, expected), f"Expected {expected}, but got {actual}"


# TODO check dicts: DISTRIBUTIONS etc
