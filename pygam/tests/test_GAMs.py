# -*- coding: utf-8 -*-

import pytest

from pygam import *


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
    assert(True)

def test_LinearGAM_uni(mcycle_X_y):
    """
    check that we can fit a Linear GAM on real, univariate data
    """
    X, y = mcycle_X_y
    gam = LinearGAM().fit(X, y)
    assert(gam._is_fitted)

def test_LinearGAM_multi(wage_X_y):
    """
    check that we can fit a Linear GAM on real, multivariate data
    """
    X, y = wage_X_y
    gam = LinearGAM().fit(X, y)
    assert(gam._is_fitted)

def test_LogisticGAM(default_X_y):
    """
    check that we can fit a Logistic GAM on real data
    """
    X, y = default_X_y
    gam = LogisticGAM().fit(X, y)
    assert(gam._is_fitted)

def test_PoissonGAM(coal_X_y):
    """
    check that we can fit a Poisson GAM on real data
    """
    X, y = coal_X_y
    gam = PoissonGAM().fit(X, y)
    assert(gam._is_fitted)

def test_InvGaussGAM(trees_X_y):
    """
    check that we can fit a InvGauss GAM on real data
    """
    X, y = trees_X_y
    gam = InvGaussGAM().fit(X, y)
    assert(gam._is_fitted)

def test_GammaGAM(trees_X_y):
    """
    check that we can fit a Gamma GAM on real data
    """
    X, y = trees_X_y
    gam = GammaGAM().fit(X, y)
    assert(gam._is_fitted)

def test_CustomGAM(trees_X_y):
    """
    check that we can fit a Custom GAM on real data
    """
    X, y = trees_X_y
    gam = GAM(distribution='gamma', link='inverse').fit(X, y)
    assert(gam._is_fitted)

def test_ExpectileGAM_uni(mcycle_X_y):
    """
    check that we can fit an Expectile GAM on real, univariate data
    """
    X, y = mcycle_X_y
    gam = ExpectileGAM().fit(X, y)
    assert(gam._is_fitted)

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
