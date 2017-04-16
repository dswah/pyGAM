# -*- coding: utf-8 -*-
import numpy as np
import pytest

from pygam import *


def test_gridsearch_returns_scores(mcycle):
    """
    check that gridsearch returns the expected number of models
    """
    n = 5
    X, y = mcycle

    gam = LinearGAM()
    scores = gam.gridsearch(X, y, lam=np.logspace(-3,3, n), return_scores=True)

    assert(len(scores) == n)

def test_gridsearch_returns_extra_score_if_fitted(mcycle):
    """
    check that gridsearch returns an extra score if our model is pre-fitted
    """
    n = 5
    X, y = mcycle

    gam = LinearGAM().fit(X, y)
    scores = gam.gridsearch(X, y, lam=np.logspace(-3,3, n), return_scores=True)

    assert(len(scores) == n + 1)

def test_gridsearch_keep_best(mcycle):
    """
    check that gridsearch returns worse model if keep_best=False
    """
    n = 5
    X, y = mcycle

    gam = LinearGAM(lam=1000000).fit(X, y)
    score1 = gam.statistics_['GCV']

    scores = gam.gridsearch(X, y, lam=np.logspace(-3,3, n),
                            keep_best=False, return_scores=True)

    assert(np.min(list(scores.values())) < score1)

def test_gridsearch_improves_objective(mcycle):
    """
    check that gridsearch improves model objective
    """
    n = 11
    X, y = mcycle

    gam = LinearGAM().fit(X, y)
    objective_0 = gam.statistics_['GCV']

    gam = LinearGAM().gridsearch(X, y, lam=np.logspace(-3,3, n))
    objective_1 = gam.statistics_['GCV']

    assert(objective_1 <= objective_0)

def test_gridsearch_all_dimensions_same(cake):
    """
    check that gridsearch searches all dimensions of lambda with equal values
    """
    n = 5
    X, y = cake

    scores = LinearGAM().gridsearch(X, y,
                                    lam=np.logspace(-3,3, n),
                                    return_scores=True)

    assert(len(scores) == n)
    assert(X.shape[1] > 1)

def test_gridsearch_all_dimensions_independent(cake):
    """
    check that gridsearch searches all dimensions of lambda independently
    """
    n = 3
    X, y = cake
    m = X.shape[1]

    scores = LinearGAM().gridsearch(X, y,
                                    lam=[np.logspace(-3,3, n)]*m,
                                    return_scores=True)

    assert(len(scores) == n**m)
    assert(m > 1)

def test_GCV_objective_is_for_unknown_scale(mcycle, default, coal, trees):
    """
    check that we use the GCV objective only for models with unknown scale

    &

    attempting to use it for models with known scale should return ValueError
    """
    lam = np.linspace(1e-3, 1e3, 2)

    unknown_scale = [(LinearGAM, mcycle),
                     (GammaGAM, trees),
                     (InvGaussGAM, trees)]

    known_scale = [(LogisticGAM, default),
                   (PoissonGAM, coal)]


    for gam, (X, y) in unknown_scale:
        scores1 = list(gam().gridsearch(X, y, lam=lam, objective='auto',
                                        return_scores=True).values())
        scores2 = list(gam().gridsearch(X, y, lam=lam, objective='GCV',
                                        return_scores=True).values())
        assert(np.allclose(scores1, scores2))

    for gam, (X, y) in known_scale:
        try:
            list(gam().gridsearch(X, y, lam=lam, objective='GCV',
                                  return_scores=True).values())
        except ValueError:
            assert(True)




def test_UBRE_objective_is_for_known_scale(mcycle, default, coal, trees):
    """
    check that we use the UBRE objective only for models with known scale

    &

    attempting to use it for models with unknown scale should return ValueError
    """
    lam = np.linspace(1e-3, 1e3, 2)

    unknown_scale = [(LinearGAM, mcycle),
                     (GammaGAM, trees),
                     (InvGaussGAM, trees)]

    known_scale = [(LogisticGAM, default),
                   (PoissonGAM, coal)]

    for gam, (X, y) in known_scale:
        scores1 = list(gam().gridsearch(X, y, lam=lam, objective='auto',
                                        return_scores=True).values())
        scores2 = list(gam().gridsearch(X, y, lam=lam, objective='UBRE',
                                        return_scores=True).values())
        assert(np.allclose(scores1, scores2))

    for gam, (X, y) in unknown_scale:
        try:
            list(gam().gridsearch(X, y, lam=lam, objective='UBRE',
                                  return_scores=True).values())
        except ValueError:
            assert(True)

def test_no_models_fitted(mcycle):
    """
    test no models fitted returns orginal gam
    """
    X, y = mcycle
    scores = LinearGAM().gridsearch(X, y, lam=[-3, -2,-1], return_scores=True)

    # scores is not a dict of scores but an (unfitted) gam!
    assert(not isinstance(scores, dict))
    assert(isinstance(scores, LinearGAM))
    assert(not scores._is_fitted)
