import numpy as np
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


# TODO check dicts: DISTRIBUTIONS etc


def test_logisticgam_predict_returns_ints(default_X_y):
    """
    check that LogisticGAM predict returns integer labels (0 or 1), not booleans
    """
    X, y = default_X_y
    gam = LogisticGAM().fit(X, y)
    preds = gam.predict(X)
    assert preds.dtype == int
    assert set(np.unique(preds)).issubset({0, 1})


def test_logisticgam_predict_proba_shape(default_X_y):
    """
    check that LogisticGAM predict_proba returns shape (n_samples, 2)
    """
    X, y = default_X_y
    gam = LogisticGAM().fit(X, y)
    probs = gam.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)
    # the probabilities should sum to 1 for each sample
    assert np.allclose(probs.sum(axis=1), np.ones(len(X)))


def test_logisticgam_has_classes(default_X_y):
    """
    check that LogisticGAM sets the classes_ attribute after fitting
    """
    X, y = default_X_y
    gam = LogisticGAM().fit(X, y)
    assert hasattr(gam, "classes_")
    assert np.array_equal(gam.classes_, np.unique(y))
