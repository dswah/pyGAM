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
    """ensure that X and y data are float and correct shapes"""
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
        X = x[:-1] + np.diff(x) / 2  # get midpoints of bins
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

    A histogram of 200 bins has been computed describing the wating time
    between eruptions.

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
        X = x[:-1] + np.diff(x) / 2  # get midpoints of bins
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
        X[:, -1] = np.unique(X[:, -1], return_inverse=True)[1]
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
        default[:, 0] = np.unique(default[:, 0], return_inverse=True)[1]
        default[:, 1] = np.unique(default[:, 1], return_inverse=True)[1]
        X = default[:, 1:]
        y = default[:, 0]
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
        X[:, 0] = np.unique(cake.values[:, 1], return_inverse=True)[1]
        X[:, 1] -= 1
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

    Source:
    Keiding, N. (1991) Age-specific incidence and prevalence: a statistical perspective
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


def toy_classification(return_X_y=True, n=5000):
    """toy classification dataset with irrelevant features

    fitting a logistic model on this data and performing a model summary
    should reveal that features 2,3,4 are not significant.

    Parameters
    ----------
    return_X_y : bool,
        if True, returns a model-ready tuple of data (X, y)
        otherwise, returns a Pandas DataFrame

    n : int, default: 5000
        number of samples to generate

    Returns
    -------
    model-ready tuple of data (X, y)
        OR
    Pandas DataFrame

    Notes
    -----
    X contains 5 variables:
        continuous feature 0
        continuous feature 1
        irrelevant feature 0
        irrelevant feature 1
        irrelevant feature 2
        categorical feature 0

    y contains binary labels

    Also, this dataset is randomly generated and will vary each time.
    """
    # make features
    X = np.random.rand(n, 5) * 10 - 5
    cat = np.random.randint(0, 4, n)
    X = np.c_[X, cat]

    # make observations
    log_odds = (
        (-0.5 * X[:, 0] ** 2) + 5 + (-0.5 * X[:, 1] ** 2) + np.mod(X[:, -1], 2) * -30
    )
    p = 1 / (1 + np.exp(-log_odds)).squeeze()
    y = (np.random.rand(n) < p).astype(int)

    if return_X_y:
        return X, y
    else:
        return pd.DataFrame(
            np.c_[X, y],
            columns=[
                [
                    'continuous0',
                    'continuous1',
                    'irrelevant0',
                    'irrelevant1',
                    'irrelevant2',
                    'categorical0',
                    'observations',
                ]
            ],
        )


def head_circumference(return_X_y=True):
    """head circumference for dutch boys

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
    X contains the age in years of each patient.

    y contains the head circumference in centimeters
    """
    # y is real
    # recommend ExpectileGAM
    head = pd.read_csv(PATH + '/head_circumference.csv', index_col=0).astype(float)
    if return_X_y:
        y = head['head'].values
        X = head[['age']].values
        return _clean_X_y(X, y)
    return head


def chicago(return_X_y=True):
    """Chicago air pollution and death rate data

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
    X contains [['time', 'tmpd', 'pm10median', 'o3median']], with no NaNs

    y contains 'death', the deaths per day, with no NaNs

    Source:
    R gamair package
    `data(chicago)`

    Notes
    -----
    https://cran.r-project.org/web/packages/gamair/gamair.pdf
    https://rdrr.io/cran/gamair/man/chicago.html

    Columns:
    death : total deaths (per day).
    pm10median : median particles in 2.5-10 per cubic m
    pm25median : median particles < 2.5 mg per cubic m (more dangerous).
    o3median : Ozone in parts per billion
    so2median : Median Sulpher dioxide measurement
    time : time in days
    tmpd : temperature in fahrenheit
    """
    # recommend PoissonGAM
    chi = pd.read_csv(PATH + '/chicago.csv', index_col=0).astype(float)
    if return_X_y:
        chi = chi[['time', 'tmpd', 'pm10median', 'o3median', 'death']].dropna()

        X = chi[['time', 'tmpd', 'pm10median', 'o3median']].values
        y = chi['death'].values

        return X, y
    else:
        return chi


def toy_interaction(return_X_y=True, n=50000, stddev=0.1):
    """a sinusoid modulated by a linear function

    this is a simple dataset to test a model's capacity to fit interactions
    between features.

    a GAM with no interaction terms will have an R-squared close to 0,
    while a GAM with a tensor product will have R-squared close to 1.

    the data is random, and will vary on each invocation.

    Parameters
    ----------
    return_X_y : bool,
        if True, returns a model-ready tuple of data (X, y)
        otherwise, returns a Pandas DataFrame

    n : int, optional
        number of points to generate

    stddev : positive float, optional,
        standard deviation of irreducible error

    Returns
    -------
    model-ready tuple of data (X, y)
        OR
    Pandas DataFrame

    Notes
    -----
    X contains [['sinusoid', 'linear']]

    y is formed by multiplying the sinusoid by the linear function.

    Source:
    """
    X = np.random.uniform(-1, 1, size=(n, 2))
    X[:, 1] *= 5

    y = np.sin(X[:, 0] * 2 * np.pi * 1.5) * X[:, 1]
    y += np.random.randn(len(X)) * stddev

    if return_X_y:
        return X, y

    else:
        data = pd.DataFrame(np.c_[X, y])
        data.columns = [['sinusoid', 'linear', 'y']]
        return data
