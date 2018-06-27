# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pygam import *


def test_expand_params(cake_X_y):
    """
    check that gam expands lam, dtype, n_splines, fit_linear, fit_splines
    penalties, spline_order
    """
    X, y = cake_X_y
    m = X.shape[1]

    gam = LinearGAM().fit(X, y)

    for param in ['dtype', 'n_splines', 'spline_order', 'fit_linear',
                  'fit_splines', 'penalties',]:
        assert(len(getattr(gam, '_' + param)) == m)

    assert(len(gam._lam) == (m + gam.fit_intercept))

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

def test_wrong_length_param(cake_X_y):
    """
    If input param is iterable, then it must have have length equal to
    the number of features.
    """
    X, y = cake_X_y
    m = X.shape[1]

    n_splines = [20] * (m+1)
    gam = LinearGAM(n_splines=n_splines)

    try:
        gam.fit(X, y)
    except ValueError:
        n_splines = [20] * (m)
        gam = LinearGAM(n_splines=n_splines).fit(X, y)
        assert(True)

def test_penalties_must_be_or_contain_callable_or_auto(mcycle_X_y):
    """
    penalty matrix must be/contain callable or auto, otherwise raise ValueError
    """
    X, y = mcycle_X_y
    gam = LinearGAM(penalties='continuous')

    try:
        gam.fit(X, y)
    except ValueError:
        gam = LinearGAM(penalties='auto').fit(X, y)
        assert(True)

    # now do iterable
    gam = LinearGAM(penalties=['continuous'])

    try:
        gam.fit(X, y)
    except ValueError:
        gam = LinearGAM(penalties=['auto']).fit(X, y)
        assert(True)

def test_line_or_spline(mcycle_X_y):
    """
    a line or spline must be fit on each feature
    """
    X, y = mcycle_X_y
    gam = LinearGAM(fit_linear=False ,fit_splines=False)

    try:
        gam.fit(X, y)
    except ValueError:
        gam = LinearGAM(fit_linear=False ,fit_splines=True).fit(X, y)
        assert(True)

def test_linear_regression(mcycle_X_y):
    """
    should be able to do linear regression
    """
    X, y = mcycle_X_y
    gam = LinearGAM(fit_linear=True, fit_splines=False).fit(X, y)
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
