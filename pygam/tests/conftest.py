# -*- coding: utf-8 -*-

import pytest
import pandas as pd
import numpy as np

from pygam import *
from pygam.datasets import (mcycle, coal, faithful, cake, coal, default, trees,
                            hepatitis, wage, toy_classification,
                            head_circumference, chicago, toy_interaction)


@pytest.fixture
def mcycle_X_y():
    # y is real
    # recommend LinearGAM
    return mcycle(return_X_y=True)

@pytest.fixture
def mcycle_gam(mcycle_X_y):
    X, y = mcycle_X_y
    gam = LinearGAM().fit(X,y)
    return gam

@pytest.fixture
def coal_X_y():
    # y is counts
    # recommend PoissonGAM
    return coal(return_X_y=True)

@pytest.fixture
def faithful_X_y():
    # y is counts
    # recommend PoissonGAM
    return faithful(return_X_y=True)

@pytest.fixture
def wage_X_y():
    # y is real
    # recommend LinearGAM
    return wage(return_X_y=True)

@pytest.fixture
def wage_gam(wage_X_y):
    X, y = wage_X_y
    gam = LinearGAM(s(0) + s(1) + f(2)).fit(X, y)
    return gam

@pytest.fixture
def trees_X_y():
    # y is real.
    # recommend InvGaussGAM, or GAM(distribution='gamma', link='log')
    return trees(return_X_y=True)

@pytest.fixture
def default_X_y():
    # y is binary
    # recommend LogisticGAM
    return default(return_X_y=True)

@pytest.fixture
def cake_X_y():
    # y is real
    # recommend LinearGAM
    return cake(return_X_y=True)

@pytest.fixture
def hepatitis_X_y():
    # y is real
    # recommend LinearGAM
    return hepatitis(return_X_y=True)

@pytest.fixture
def toy_classification_X_y():
    # y is binary ints
    # recommend LogisticGAM
    return toy_classification(return_X_y=True)

@pytest.fixture
def head_circumference_X_y():
    # centile data
    # recommend ExpectileGAM
    return head_circumference(return_X_y=True)

@pytest.fixture
def chicago_X_y():
    # y is counts
    # recommend PoissonGAM
    return chicago(return_X_y=True)

@pytest.fixture
def chicago_gam(chicago_X_y):
    X, y = chicago_X_y
    gam = PoissonGAM(terms=s(0, n_splines=200) + te(3, 1) + s(2)).fit(X, y)
    return gam

@pytest.fixture
def toy_interaction_X_y():
    # y is real
    # recommend LinearGAM
    return toy_interaction(return_X_y=True)
