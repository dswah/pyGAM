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
    assert(True)

def test_LinearGAM_uni(mcycle):
    """
    check that we can fit a Linear GAM on real, univariate data
    """
    X, y = mcycle
    gam = LinearGAM().fit(X, y)
    assert(gam._is_fitted)

def test_LinearGAM_multi(wage):
    """
    check that we can fit a Linear GAM on real, multivariate data
    """
    X, y = wage
    gam = LinearGAM().fit(X, y)
    assert(gam._is_fitted)

def test_LogisticGAM(default):
    """
    check that we can fit a Logistic GAM on real data
    """
    X, y = default
    gam = LogisticGAM().fit(X, y)
    assert(gam._is_fitted)

def test_PoissonGAM(coal):
    """
    check that we can fit a Poisson GAM on real data
    """
    X, y = coal
    gam = PoissonGAM().fit(X, y)
    assert(gam._is_fitted)

def test_InvGaussGAM(trees):
    """
    check that we can fit a InvGauss GAM on real data
    """
    X, y = trees
    gam = InvGaussGAM().fit(X, y)
    assert(gam._is_fitted)

def test_GammaGAM(trees):
    """
    check that we can fit a Gamma GAM on real data
    """
    X, y = trees
    gam = GammaGAM().fit(X, y)
    assert(gam._is_fitted)

def test_CustomGAM(trees):
    """
    check that we can fit a Custom GAM on real data
    """
    X, y = trees
    gam = GAM(distribution='gamma', link='log').fit(X, y)
    assert(gam._is_fitted)

# TODO check dicts: DISTRIBUTIONS etc
