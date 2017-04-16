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

# test auto objective for known scale is UBRE, GCV for unknown
