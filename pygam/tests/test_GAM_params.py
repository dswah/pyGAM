# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pygam import *


def test_lam_non_neg_array_like(cake_X_y):
    """
    lambda must be a non-negative float or array of floats
    """
    X, y = cake_X_y

    try:
        gam = LinearGAM(lam=-1).fit(X, y)
    except ValueError:
        assert(True)

    try:
        gam = LinearGAM(lam=['hi']).fit(X, y)
    except ValueError:
        assert(True)

def test_penalties_must_be_or_contain_callable_or_auto(mcycle_X_y):
    """
    penalty matrix must be/contain callable or auto, otherwise raise ValueError
    """
    X, y = mcycle_X_y

    with pytest.raises(ValueError):
        gam = LinearGAM(terms=s(0, penalties='continuous'))

    # now do iterable
    with pytest.raises(ValueError):
        gam = LinearGAM(s(0, penalties=['continuous']))

def test_intercept(mcycle_X_y):
    """
    should be able to just fit intercept
    """
    X, y = mcycle_X_y
    gam = LinearGAM(terms=intercept)
    gam.fit(X, y)

def test_require_one_term(mcycle_X_y):
    """
    need at least one term
    """
    X, y = mcycle_X_y
    gam = LinearGAM(terms=[])
    with pytest.raises(ValueError):
        gam.fit(X, y)

def test_linear_regression(mcycle_X_y):
    """
    should be able to do linear regression
    """
    X, y = mcycle_X_y
    gam = LinearGAM(l(0)).fit(X, y)
    assert(gam._is_fitted)

def test_compute_stats_even_if_not_enough_iters(default_X_y):
    """
    should be able to do linear regression
    """
    X, y = default_X_y
    gam = LogisticGAM(max_iter=1).fit(X, y)
    assert(hasattr(gam, 'statistics_'))

# TODO categorical dtypes get no fit linear even if fit linear TRUE
# TODO categorical dtypes get their own number of splines
# TODO can force continuous dtypes on categorical vars if wanted
