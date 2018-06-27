"""
GAM datasets
"""
# -*- coding: utf-8 -*-

from os.path import dirname

import pandas as pd
import numpy as np

from pygam.utils import make_2d


PATH = dirname(__file__)


def _clean_X_y(X, y):
    """ensure that X and y data are float and correct shapes
    """
    return make_2d(X, verbose=False).astype('float'), y.astype('float')

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
    X contains the times after the impact.
    y contains the acceleration.

    Source:
    https://vincentarelbundock.github.io/Rdatasets/doc/MASS/mcycle.html
    """
    # y is real
    # recommend LinearGAM
    motor = pd.read_csv(PATH + '/mcycle.csv', index_col=0)
    if return_X_y:
        X = motor.times.values
        y = motor.accel
        return _clean_X_y(X, y)
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
    The (X, y) tuple is a processed version of the otherwise raw DataFrame.

    A histogram of 150 bins has been computed describing the number accidents per year.

    X contains the midpoints of histogram bins.
    y contains the count in each histogram bin.

    Source:
    https://vincentarelbundock.github.io/Rdatasets/doc/boot/coal.html
    """
    # y is counts
    # recommend PoissonGAM
    coal = pd.read_csv(PATH + '/coal.csv', index_col=0)
    if return_X_y:
        y, x = np.histogram(coal.values, bins=150)
        X = x[:-1] + np.diff(x)/2 # get midpoints of bins
        return _clean_X_y(X, y)
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
    The (X, y) tuple is a processed version of the otherwise raw DataFrame.

    A histogram of 200 bins has been computed describing the wating time between eruptions.

    X contains the midpoints of histogram bins.
    y contains the count in each histogram bin.

    Source:
    https://vincentarelbundock.github.io/Rdatasets/doc/datasets/faithful.html
    """
    # y is counts
    # recommend PoissonGAM
    faithful = pd.read_csv(PATH + '/faithful.csv', index_col=0)
    if return_X_y:
        y, x = np.histogram(faithful['eruptions'], bins=200)
        X = x[:-1] + np.diff(x)/2 # get midpoints of bins
        return _clean_X_y(X, y)
    return faithful

def wage(return_X_y=True):
    """wage dataset

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
    X contains the year, age and education of each sampled person.
    The education category has been transformed to integers.

    y contains the wage.

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
        return _clean_X_y(X, y)
    return wage

def trees(return_X_y=True):
    """cherry trees dataset

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
    X contains the girth and the height of each tree.
    y contains the volume.

    Source:
    https://vincentarelbundock.github.io/Rdatasets/doc/datasets/trees.html
    """
    # y is real.
    # recommend InvGaussGAM, or GAM(distribution='gamma', link='log')
    trees = pd.read_csv(PATH + '/trees.csv', index_col=0)
    if return_X_y:
        y = trees.Volume.values
        X = trees[['Girth', 'Height']].values
        return _clean_X_y(X, y)
    return trees

def default(return_X_y=True):
    """credit default dataset

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
    X contains the category of student or not, credit card balance,
    and  income.

    y contains the outcome of default (0) or not (1).

    Source:
    https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/Default.html
    """
    # y is binary
    # recommend LogisticGAM
    default = pd.read_csv(PATH + '/default.csv', index_col=0)
    if return_X_y:
        default = default.values
        default[:,0] = np.unique(default[:,0], return_inverse=True)[1]
        default[:,1] = np.unique(default[:,1], return_inverse=True)[1]
        X = default[:,1:]
        y = default[:,0]
        return _clean_X_y(X, y)
    return default

def cake(return_X_y=True):
    """cake dataset

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
    X contains the category of recipe used transformed to an integer,
    the catergory of replicate, and the temperatue.

    y contains the angle at which the cake broke.

    Source:
    https://vincentarelbundock.github.io/Rdatasets/doc/lme4/cake.html
    """
    # y is real
    # recommend LinearGAM
    cake = pd.read_csv(PATH + '/cake.csv', index_col=0)
    if return_X_y:
        X = cake[['recipe', 'replicate', 'temperature']].values
        X[:,0] = np.unique(cake.values[:,1], return_inverse=True)[1]
        X[:,1] -= 1
        y = cake['angle'].values
        return _clean_X_y(X, y)
    return cake

def hepatitis(return_X_y=True):
    """hepatitis in Bulgaria dataset

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
    X contains the age of each patient group.

    y contains the ratio of HAV positive patients to the total number for each
    age group.

    Groups with 0 total patients are excluded.
    """
    # y is real
    # recommend LinearGAM
    hep = pd.read_csv(PATH + '/hepatitis_A_bulgaria.csv').astype(float)
    if return_X_y:
        # eliminate 0/0
        mask = (hep.total > 0).values
        hep = hep[mask]

        X = hep.age.values
        y = hep.hepatitis_A_positive.values / hep.total.values
        return _clean_X_y(X, y)
    return hep
