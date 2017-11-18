# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pygam import *
from pygam.utils import check_X, check_y, check_X_y


# TODO check dtypes works as expected
# TODO checkX, checky, check XY expand as needed, call out bad domain

@pytest.fixture
def wage_gam(wage):
    X, y = wage
    gam = LinearGAM().fit(X,y)
    return gam

@pytest.fixture
def default_gam(default):
    X, y = default
    gam = LogisticGAM().fit(X,y)
    return gam

def test_check_X_categorical_prediction_exceeds_training(wage, wage_gam):
    """
    if our categorical variable is outside the training range
    we should get an error
    """
    X, y = wage
    gam = wage_gam
    eks = gam._edge_knots[-1] # last feature of wage dataset is categorical
    X[:,-1] = eks[-1] + 1

    try:
        gam.predict(X)
        assert False
    except ValueError:
        X[:,-1] = eks[-1]
        gam.predict(X)
        assert(True)

def test_check_y_not_int_not_float(wage, wage_gam):
    """y must be int or float, or we should get a value error"""
    X, y = wage
    y_str = ['hi'] * len(y)
    try:
        check_y(y_str, wage_gam.link, wage_gam.distribution)
        assert False
    except ValueError:
        check_y(y, wage_gam.link, wage_gam.distribution)
        assert(True)

def test_check_y_casts_to_numerical(wage, wage_gam):
    """check_y will try to cast data to numerical types"""
    X, y = wage
    y = y.astype('object')

    y = check_y(y, wage_gam.link, wage_gam.distribution)
    assert y.dtype == 'float'


def test_check_y_not_min_samples(wage, wage_gam):
    """check_y expects a minimum number of samples"""
    X, y = wage
    try:
        check_y(y, wage_gam.link, wage_gam.distribution, min_samples=len(y)+1)
        assert False
    except ValueError:
        check_y(y, wage_gam.link, wage_gam.distribution, min_samples=len(y))
        assert True

def test_check_y_not_in_doamin_link(default, default_gam):
    """if you give labels outide of the links domain, check_y will raise an error"""
    X, y = default
    gam = default_gam

    try:
        check_y(y + .1, default_gam.link, default_gam.distribution)
        assert False
    except ValueError:
        check_y(y, default_gam.link, default_gam.distribution)
        assert True

def test_check_X_not_int_not_float():
    """X  must be an in or a float"""

    try:
        check_X(['hi'])
        assert False
    except ValueError:
        check_X([4])
        assert True

def test_check_X_too_many_dims():
    """check_X accepts at most 2D inputs"""
    try:
        check_X(np.ones((5,4,3)))
        assert False
    except ValueError:
        check_X(np.ones((5,4)))
        assert True

def test_check_X_not_min_samples():
    try:
        check_X(np.ones((5)), min_samples=6)
        assert False
    except ValueError:
        check_X(np.ones((5)), min_samples=5)
        assert True

def test_check_X_y_different_lengths():
    try:
        check_X_y(np.ones(5), np.ones(4))
        assert False
    except ValueError:
        check_X_y(np.ones(5), np.ones(5))
        assert True
# # def test_b_spline_basis_clamped_what_we_want():
