"""
GAM datasets
"""
# -*- coding: utf-8 -*-

from os.path import dirname

import pandas as pd
import numpy as np


PATH = dirname(__file__)


def mcycle(return_X_y=True):
    """motorcyle acceleration dataset

    Parameters
    ----------
    return_X_y : bool,
        if True, returns a model-ready tuple of data (X, y)
        otherwise, returns a Pandas DataFrame

    Returns
    -------
    model-ready tuple of data (X, y)
        OR
    Pandas DataFrame

    Notes
    -----
    Source:
    https://vincentarelbundock.github.io/Rdatasets/doc/MASS/mcycle.html
    """
    # y is real
    # recommend LinearGAM
    motor = pd.read_csv(PATH + '/mcycle.csv', index_col=0)
    if return_X_y:
        X = motor.times.values
        y = motor.accel
        return X, y
    return motor

def coal(return_X_y=True):
    """coal-mining accidents dataset

    Parameters
    ----------
    return_X_y : bool,
        if True, returns a model-ready tuple of data (X, y)
        otherwise, returns a Pandas DataFrame

    Returns
    -------
    model-ready tuple of data (X, y)
        OR
    Pandas DataFrame

    Notes
    -----
    Source: https://vincentarelbundock.github.io/Rdatasets/doc/boot/coal.html

    The (X, y) tuple is a processed version of the otherwise raw DataFrame.

    A histogram has been computed describing the number accidents per year.
    X contains the midpoints of histogram bins,
    y contains the count in each histogram bin.
    """
    # y is counts
    # recommend PoissonGAM
    coal = pd.read_csv(PATH + '/coal.csv', index_col=0)
    if return_X_y:
        y, x = np.histogram(coal.values, bins=150)
        X = x[:-1] + np.diff(x)/2 # get midpoints of bins
        return X, y
    return coal

def faithful(return_X_y=True):
    """old-faithful dataset

    Parameters
    ----------
    return_X_y : bool,
        if True, returns a model-ready tuple of data (X, y)
        otherwise, returns a Pandas DataFrame

    Returns
    -------
    model-ready tuple of data (X, y)
        OR
    Pandas DataFrame

    Notes
    -----
    Source:
    https://vincentarelbundock.github.io/Rdatasets/doc/datasets/faithful.html

    The (X, y) tuple is a processed version of the otherwise raw DataFrame.

    A histogram has been computed describing the wating time between eruptions.
    X contains the midpoints of histogram bins,
    y contains the count in each histogram bin.
    """
    # y is counts
    # recommend PoissonGAM
    faithful = pd.read_csv(PATH + '/faithful.csv', index_col=0)
    if return_X_y:
        y, x = np.histogram(faithful.values, bins=200)
        X = x[:-1] + np.diff(x)/2 # get midpoints of bins
        return X, y
    return faithful

def wage(return_X_y=True):
    """wagel dataset

    Parameters
    ----------
    return_X_y : bool,
        if True, returns a model-ready tuple of data (X, y)
        otherwise, returns a Pandas DataFrame

    Returns
    -------
    model-ready tuple of data (X, y)
        OR
    Pandas DataFrame

    Notes
    -----

    Source:
    https://github.com/JWarmenhoven/ISLR-python/blob/master/Notebooks/Data/Wage.csv
    """
    # y is real
    # recommend LinearGAM
    wage = pd.read_csv(PATH + '/wage.csv', index_col=0)
    if return_X_y:
        X = wage[['year', 'age', 'education']].values
        X[:,-1] = np.unique(X[:,-1], return_inverse=True)[1]
        y = wage['wage'].values
        return X, y
    return wage

def trees(return_X_y=True):
    # y is real.
    # recommend InvGaussGAM, or GAM(distribution='gamma', link='log')
    trees = pd.read_csv(PATH + '/trees.csv', index_col=0)
    if return_X_y:
        y = trees.Volume.values
        X = trees[['Girth', 'Height']].values
        return X, y
    return trees

def default(return_X_y=True):
    # y is binary
    # recommend LogisticGAM
    default = pd.read_csv(PATH + '/default.csv', index_col=0)
    if return_X_y:
        default = default.values
        default[:,0] = np.unique(default[:,0], return_inverse=True)[1]
        default[:,1] = np.unique(default[:,1], return_inverse=True)[1]
        X = default[:,1:]
        y = default[:,0]
        return X, y
    return default

def cake(return_X_y=True):
    # y is real
    # recommend LinearGAM
    cake = pd.read_csv(PATH + '/cake.csv', index_col=0)
    if return_X_y:
        X = cake[['recipe', 'replicate', 'temperature']].values
        X[:,0] = np.unique(cake.values[:,1], return_inverse=True)[1]
        X[:,1] -= 1
        y = cake['angle'].values
        return X, y
    return cake

def hepatitis(return_X_y=True):
    # y is real
    # recommend LinearGAM
    hep = pd.read_csv(PATH + '/hepatitis_A_bulgaria.csv').astype(float)
    if return_X_y:
        # eliminate 0/0
        mask = (hep.total > 0).values
        hep = hep[mask]

        X = hep.age.values
        y = hep.hepatitis_A_positive.values / hep.total.values
        return X, y
    return hep
