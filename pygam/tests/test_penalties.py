# -*- coding: utf-8 -*-

import numpy as np

from pygam import LinearGAM, s

from pygam.penalties import derivative
from pygam.penalties import l2
from pygam.penalties import monotonic_inc
from pygam.penalties import monotonic_dec
from pygam.penalties import convex
from pygam.penalties import concave
from pygam.penalties import none
from pygam.penalties import wrap_penalty


def test_single_spline_penalty():
    """
    check that feature functions with only 1 basis are penalized correctly

    derivative penalty should be 0.
    l2 should penalty be 1.
    monotonic_ and convexity_ should be 0.
    """
    coef = np.array(1.0)
    assert np.alltrue(derivative(1, coef).A == 0.0)
    assert np.alltrue(l2(1, coef).A == 1.0)
    assert np.alltrue(monotonic_inc(1, coef).A == 0.0)
    assert np.alltrue(monotonic_dec(1, coef).A == 0.0)
    assert np.alltrue(convex(1, coef).A == 0.0)
    assert np.alltrue(concave(1, coef).A == 0.0)
    assert np.alltrue(none(1, coef).A == 0.0)


def test_wrap_penalty():
    """
    check that wrap penalty indeed reduces inserts the desired penalty into the
    linear term when fit_linear is True, and 0, when fit_linear is False.
    """
    coef = np.array(1.0)
    n = 2
    linear_penalty = -1

    fit_linear = True
    p = wrap_penalty(none, fit_linear, linear_penalty=linear_penalty)
    P = p(n, coef).A
    assert P.sum() == linear_penalty

    fit_linear = False
    p = wrap_penalty(none, fit_linear, linear_penalty=linear_penalty)
    P = p(n, coef).A
    assert P.sum() == 0.0


def test_monotonic_inchepatitis_X_y(hepatitis_X_y):
    """
    check that monotonic_inc constraint produces monotonic increasing function
    """
    X, y = hepatitis_X_y

    gam = LinearGAM(terms=s(0, constraints='monotonic_inc'))
    gam.fit(X, y)

    XX = gam.generate_X_grid(term=0)
    Y = gam.predict(np.sort(XX))
    diffs = np.diff(Y, n=1)
    assert ((diffs >= 0) + np.isclose(diffs, 0.0)).all()


def test_monotonic_dec(hepatitis_X_y):
    """
    check that monotonic_dec constraint produces monotonic decreasing function
    """
    X, y = hepatitis_X_y

    gam = LinearGAM(terms=s(0, constraints='monotonic_dec'))
    gam.fit(X, y)

    XX = gam.generate_X_grid(term=0)
    Y = gam.predict(np.sort(XX))
    diffs = np.diff(Y, n=1)
    assert ((diffs <= 0) + np.isclose(diffs, 0.0)).all()


def test_convex(hepatitis_X_y):
    """
    check that convex constraint produces convex function
    """
    X, y = hepatitis_X_y

    gam = LinearGAM(terms=s(0, constraints='convex'))
    gam.fit(X, y)

    XX = gam.generate_X_grid(term=0)
    Y = gam.predict(np.sort(XX))
    diffs = np.diff(Y, n=2)
    assert ((diffs >= 0) + np.isclose(diffs, 0.0)).all()


def test_concave(hepatitis_X_y):
    """
    check that concave constraint produces concave function
    """
    X, y = hepatitis_X_y

    gam = LinearGAM(terms=s(0, constraints='concave'))
    gam.fit(X, y)

    XX = gam.generate_X_grid(term=0)
    Y = gam.predict(np.sort(XX))
    diffs = np.diff(Y, n=2)
    assert ((diffs <= 0) + np.isclose(diffs, 0.0)).all()


# TODO penalties gives expected matrix structure
# TODO circular constraints
