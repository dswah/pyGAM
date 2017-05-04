# -*- coding: utf-8 -*-

import pytest
import pandas as pd
import numpy as np

from pygam import *


@pytest.fixture
def mcycle():
    # y is real
    # recommend LinearGAM
    motor = pd.read_csv('datasets/mcycle.csv', index_col=0)
    X = motor.times.values
    y = motor.accel
    return X, y

@pytest.fixture
def coal():
    # y is counts
    # recommend PoissonGAM
    coal = pd.read_csv('datasets/coal.csv', index_col=0)
    y, x = np.histogram(coal.values, bins=150)
    X = x[:-1] + np.diff(x)/2 # get midpoints of bins
    return X, y

@pytest.fixture
def faithful():
    # y is counts
    # recommend PoissonGAM
    faithful = pd.read_csv('datasets/faithful.csv', index_col=0)
    y, x = np.histogram(faithful.values, bins=200)
    X = x[:-1] + np.diff(x)/2 # get midpoints of bins
    return X, y

@pytest.fixture
def wage():
    # y is real
    # recommend LinearGAM
    wage = pd.read_csv('datasets/wage.csv', index_col=0)
    X = wage[['year', 'age', 'education']].values
    X[:,-1] = np.unique(X[:,-1], return_inverse=True)[1]
    y = wage['wage'].values
    return X, y

@pytest.fixture
def trees():
    # y is real.
    # recommend InvGaussGAM, or GAM(distribution='gamma', link='log')
    trees = pd.read_csv('datasets/trees.csv', index_col=0)
    y = trees.Volume.values
    X = trees[['Girth', 'Height']].values
    return X, y

@pytest.fixture
def default():
    # y is binary
    # recommend LogisticGAM
    default = pd.read_csv('datasets/default.csv', index_col=0)
    default = default.values
    default[:,0] = np.unique(default[:,0], return_inverse=True)[1]
    default[:,1] = np.unique(default[:,1], return_inverse=True)[1]
    X = default[:,1:]
    y = default[:,0]
    return X, y

@pytest.fixture
def cake():
    # y is real
    # recommend LinearGAM
    cake = pd.read_csv('datasets/cake.csv', index_col=0)
    X = cake[['recipe', 'replicate', 'temperature']].values
    X[:,0] = np.unique(cake.values[:,1], return_inverse=True)[1]
    X[:,1] -= 1
    y = cake['angle'].values
    return X, y

@pytest.fixture
def hepatitis():
    # y is real
    # recommend LinearGAM
    hep = pd.read_csv('datasets/hepatitis_A_bulgaria.csv').astype(float)

    # eliminate 0/0
    mask = (hep.total > 0).values
    hep = hep[mask]

    X = hep.age.values
    y = hep.hepatitis_A_positive.values / hep.total.values
    return X, y
