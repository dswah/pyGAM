"""
Pygam utilities
"""

import scipy as sp
from scipy import sparse

import numpy as np


def check_dtype_(X):
    jitter = np.random.randn(X.shape[0])
    dtypes_ = []
    for feat in X.T:
        dtype = feat.dtype.type
        assert issubclass(dtype, (np.int, np.float)), 'data must be discrete or continuous valued'

        if issubclass(dtype, np.int) or (len(np.unique(feat)) != len(np.unique(feat + jitter))):
            assert (np.max(feat) - np.min(feat)) == (len(np.unique(feat)) - 1), 'k categories must be mapped to integers in [0, k-1] interval'
            dtypes_.append(np.int)
            continue

        if issubclass(dtype, np.float):
            dtypes_.append(np.float)
            continue
    return dtypes_


def check_y(y, link, dist):
    y = np.ravel(y)
    assert np.all(~(np.isnan(link.link(y, dist)))), 'y data is not in domain of link function'
    return y


def round_to_n_decimal_places(array, n=3):
    """
    tool to keep round a float to n decimal places.

    n=3 by default
    """
    # check if in scientific notation
    if issubclass(array.__class__, float) and '%.e'%array == str(array):
        return array # do nothing

    shape = np.shape(array)
    return ((np.atleast_1d(array) * 10**n).round().astype('int') / (10.**n)).reshape(shape)


def print_data(data_dict, width=-5, keep_decimals=3, fill=' ', title=None):
    """
    tool to print a dictionary with a nice formatting

    Parameters:
    -----------
    data_dict:
        dict. Dictionary to be printed.
    width:
        int. Desired total line width.
        A negative value will fill to minimum required width + neg(width)
        default: -5
    keep_decimals:
        int. number of decimal places to keep:
        default: 3
    fill:
        string. the character to fill between keys and values.
        Must have length 1.
        default: ' '
    title:
        string.
        default: None
    """

    # find max length
    keys = np.array(data_dict.keys(), dtype='str')
    values = round_to_n_decimal_places(np.array(data_dict.values())).astype('str')
    M = max([len(k + v) for k, v in zip(keys, values)])

    if width < 0:
        # this is for a dynamic filling.
        # fill to minimum required width + neg(width)
        width = M - width

    assert M < width, 'desired width is {}, but max data length is {}'.format(width, M)

    fill = str(fill)
    assert len(fill) == 1, 'fill must contain exactly one symbol'

    if title is not None:
        print(title)
        print('-' * width)
    for k, v in zip(keys, values):
        nk = len(k)
        nv = len(v)
        filler = fill*(width - nk - nv)
        print(k + filler + v)


def gen_knots(data, dtype, n_knots=10, add_boundaries=False):
        """
        generate knots from data quantiles

        for discrete data, assumes k categories in [0, k-1] interval
        """
        assert dtype in [np.int, np.float], 'unsupported dtype'
        if dtype == np.int:
            knots = np.r_[np.min(data) - 0.5, np.unique(data) + 0.5]
        else:
            knots = np.percentile(data, np.linspace(0,100, n_knots+2))

        if add_boundaries:
            return knots
        return knots[1:-1]


def b_spline_basis(x, boundary_knots, order=4, sparse=True):
    """
    generate b-spline basis using De Boor recursion
    """
    x = np.atleast_2d(x).T
    aug_knots = np.r_[boundary_knots.min() * np.ones(order-1), np.sort(boundary_knots), boundary_knots.max() * np.ones(order-1)]

    bases = (x >= aug_knots[:-1]).astype(np.int) * (x < aug_knots[1:]).astype(np.int) # haar bases
    bases[(x >= aug_knots[-1])[:,0], -order] = 1 # want the last basis function extend past the boundary
    bases[(x < aug_knots[0])[:,0], order] = 1

    maxi = len(aug_knots) - 1

    # do recursion from Hastie et al.
    for m in range(2, order + 1):
        maxi -= 1
        maskleft = aug_knots[m-1:maxi+m-1] == aug_knots[:maxi] # bookkeeping to avoid div by 0
        maskright = aug_knots[m:maxi+m] == aug_knots[1:maxi+1]

        left = ((x - aug_knots[:maxi]) / (aug_knots[m-1:maxi+m-1] - aug_knots[:maxi])) * bases[:,:maxi]
        left[:,maskleft] = 0.

        right = ((aug_knots[m:maxi+m]-x) / (aug_knots[m:maxi+m] - aug_knots[1:maxi+1])) * bases[:,1:maxi+1]
        right[:,maskright] = 0.

        bases = left + right

    if sparse:
        return sp.sparse.csc_matrix(bases)

    return bases


def ylogydu(y, u):
    """tool to give desired output for the limit as y -> 0, which is 0"""
    mask = (np.atleast_1d(y)!=0.)
    out = np.zeros_like(u)
    out[mask] = y[mask] * np.log(y[mask] / u[mask])
    return out
