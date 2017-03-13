"""
Pygam utilities
"""

from copy import deepcopy

import scipy as sp
from scipy import sparse

import numpy as np


def check_dtype(X):
    """
    tool to identify the data-types of the features in data matrix X.
    checks for float and int data-types.

    Parameters
    ----------
    X :
      array of shape (n_samples, n_features)

    Returns
    -------
    dtypes :
      list of types of length n_features
    """
    jitter = np.random.randn(X.shape[0])
    dtypes = []
    for feat in X.T:
        dtype = feat.dtype.type
        assert issubclass(dtype, (np.int, np.float)), 'data must be discrete or continuous valued'

        if issubclass(dtype, np.int) or (len(np.unique(feat)) != len(np.unique(feat + jitter))):
            assert (np.max(feat) - np.min(feat)) == (len(np.unique(feat)) - 1), 'k categories must be mapped to integers in [0, k-1] interval'
            dtypes.append(np.int)
            continue

        if issubclass(dtype, np.float):
            dtypes.append(np.float)
            continue
    return dtypes


def check_y(y, link, dist):
    """
    tool to ensure that the targets are in the domain of the link function

    Parameters
    ----------
    y : array-like
    link : Link object
    dist : Distribution object

    Returns
    -------
    y : array containing validated y-data
    """
    y = np.ravel(y)
    assert np.all(~np.isnan(link.link(y, dist))), \
           'y data is not in domain of {} link function. ' \
           'Expected domain: {}, but found {}' \
           .format(link,
                   get_link_domain(link, dist),
                   [float('%.2f'%np.min(y)), float('%.2f'%np.max(y))])
    return y

def check_X(X, max_feats=None):
    """
    tool to ensure that X is 2 dimensional

    Parameters
    ----------
    X : array-like
    n_coeffs : int. represents maximum number of features
               default: None

    Returns
    -------
    X : array with ndims == 2 containing validated X-data
    """
    X = np.atleast_2d(X)
    assert X.ndims <= 2, 'X must be a matrix or vector. found shape {}'.format(X.shape)
    if max_feats is not None:
        assert X.shape[1] <= max_feats, 'X data must have less than {} features, but found {}'.format(max_feats, X.shape[1])
    return X

def check_X_y(X, y):
    """
    tool to ensure input and output data have the same number of samples

    Parameters
    ----------
    X : array-like
    y : array-like

    Returns
    -------
    None
    """
    assert len(X) == len(y), 'Inconsistent input and output data shapes: found {} and {}'.format(len(X), len(y))

def get_link_domain(link, dist):
    """
    tool to identify the domain of a given monotonic link function

    Parameters
    ----------
    link : Link object
    dist : Distribution object

    Returns
    -------
    domain : list of length 2, representing the interval of the domain.
    """
    domain = np.array([-np.inf, -1, 0, 1, np.inf])
    domain = domain[~np.isnan(link.link(domain, dist))]
    return [domain[0], domain[-1]]

def round_to_n_decimal_places(array, n=3):
    """
    tool to keep round a float to n decimal places.

    n=3 by default

    Parameters
    ----------
    array : np.array
    n : int. number of decimal places to keep

    Returns
    -------
    array : rounded np.array
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

    Returns
    -------
    None
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
    tool to generate b-spline basis using vectorized De Boor recursion
    the basis functions extrapolate linearly past the end-knots.

    Parameters
    ----------
    x : array-like, with ndims == 1.
    boundary_knots : array-like contaning locations of knots, including the 2 edge knots.
    order : int. order of spline basis to create
            default: 4
    sparse : boolean. whether to return a sparse basis matrix or not.
             default: True

    Returns
    -------
    basis : sparse csc matrix or array containing b-spline basis functions of order: order
            default: sparse csc matrix
    """
    assert np.ravel(x).ndim == 1, 'data must be 1-D, but found {}'.format(np.ravel(x).ndim)

    # rescale boundary_knots to [0,1]
    boundary_knots = np.sort(deepcopy(boundary_knots))
    offset = boundary_knots[0]
    scale = boundary_knots[-1] - boundary_knots[0]
    boundary_knots -= offset
    boundary_knots /= scale

    # rescale x as well
    x = (np.ravel(deepcopy(x)) - offset) / scale
    x_extrapolte_left = (x < 0)
    x_extrapolte_right = (x > 1)
    x_interpolate = ~(x_extrapolte_right + x_extrapolte_left)
    x = np.atleast_2d(x).T
    n = len(x)

    # augment knots
    aug_knots = np.r_[boundary_knots.min() * np.ones(order-1), np.sort(boundary_knots), boundary_knots.max() * np.ones(order-1)]

    # prepare Haar Basis
    bases = (x >= aug_knots[:-1]).astype(np.int) * (x < aug_knots[1:]).astype(np.int) # haar bases
    bases[(x >= aug_knots[-1])[:,0], -order] = 1 # want the last basis function extend past the boundary
    bases[(x < aug_knots[0])[:,0], order] = 1

    maxi = len(aug_knots) - 1

    # do recursion from Hastie et al.
    for m in range(2, order + 1):
        maxi -= 1

        # bookkeeping to avoid div by 0
        maskleft = aug_knots[m-1:maxi+m-1] != aug_knots[:maxi]
        maskright = aug_knots[m:maxi+m] != aug_knots[1:maxi+1]

        # left sub-basis
        num = (x - aug_knots[:maxi][maskleft]) * bases[:,:maxi][:,maskleft]
        denom = aug_knots[m-1:maxi+m-1][maskleft] - aug_knots[:maxi][maskleft]
        left = np.zeros((n, maxi))
        left[:, maskleft] = num/denom

        # right sub-basis
        num = (aug_knots[m:maxi+m][maskright]-x) * bases[:,1:maxi+1][:,maskright]
        denom = aug_knots[m:maxi+m][maskright] - aug_knots[1:maxi+1][maskright]
        right = np.zeros((n, maxi))
        right[:, maskright] = num/denom

        bases = left + right

    # extrapolate
    # since we have repeated end-knots, only the last 2 basis functions are
    # non-zero at the end-knots, and they have equal and opposite gradient.
    if any(x_extrapolte_right) or any(x_extrapolte_left):
        bases[~x_interpolate] = 0.
        if any(x_extrapolte_left):
            grad_left = -1/(boundary_knots[1] - boundary_knots[0]) * (order-1)
            bases[x_extrapolte_left, :1] = grad_left * x[x_extrapolte_left] + 1
            bases[x_extrapolte_left, 1:2] = -grad_left * x[x_extrapolte_left]
        if any(x_extrapolte_right):
            grad_right = -1/(boundary_knots[-2] - boundary_knots[-1]) * (order-1)
            bases[x_extrapolte_right, -1:] = grad_right * (x[x_extrapolte_right] - 1) + 1
            bases[x_extrapolte_right, -2:-1] = -grad_right * (x[x_extrapolte_right] - 1)

    if sparse:
        return sp.sparse.csc_matrix(bases)

    return bases


def ylogydu(y, u):
    """tool to give desired output for the limit as y -> 0, which is 0"""
    mask = (np.atleast_1d(y)!=0.)
    out = np.zeros_like(u)
    out[mask] = y[mask] * np.log(y[mask] / u[mask])
    return out


def combine(*args):
    """
    tool to perform tree search via recursion
    useful for developing the grid in a grid search

    Parameters
    ----------
    args : list of lists

    Returns
    -------
    list of all the combinations of the elements in the input lists
    """
    if hasattr(args, '__iter__') and (len(args) > 1):
        subtree = combine(*args[:-1])
        tree = []
        for leaf in subtree:
            for node in args[-1]:
                if hasattr(leaf, '__iter__'):
                    tree.append(leaf + [node])
                else:
                    tree.append([leaf] + [node])
        return tree
    else:
        return args[0]
