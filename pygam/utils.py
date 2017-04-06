"""
Pygam utilities
"""

from __future__ import division
from copy import deepcopy
import warnings

import scipy as sp
from scipy import sparse
import numpy as np

try:
  from sksparse.cholmod import cholesky as spcholesky
  SKSPIMPORT = True
except:
  msg = 'Could not import Scikit-Sparse.\nThis will slow down optimization '\
        'for models with monotonicity/convexity penalties and many splines.\n'\
        'See installation instructions for installing Scikit-Sparse via Conda.'
  warnings.warn(msg)
  SKSPIMPORT = False


def cholesky(A, sparse=True):
    if SKSPIMPORT:
        A = sp.sparse.csc_matrix(A)

        F = spcholesky(A)

        # permutation matrix P
        P = sp.sparse.lil_matrix(A.shape)
        p = F.P()
        P[np.arange(len(p)), p] = 1

        # permute
        L = F.L()
        L = P.T.dot(L)

        if sparse:
            return L
        return L.todense()

    else:
        if sp.sparse.issparse(A):
            A = A.todense()
        L = np.linalg.cholesky(A)

        if sparse:
            return sp.sparse.csc_matrix(L)
        return L


def generate_X_grid(gam, n=500):
    """
    tool to create a nice grid of X data if no X data is supplied
    """
    X = []
    for ek in gam._edge_knots:
        X.append(np.linspace(ek[0], ek[-1], num=n))
    return np.vstack(X).T


def check_dtype(X, ratio=.95):
    """
    tool to identify the data-types of the features in data matrix X.
    checks for float and int data-types.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)

    ratio : float in [0, 1], default: 0.95
      minimum ratio of unique values to samples before a feature is considered
      categorical.

    Returns
    -------
    dtypes : list of types of length n_features
    """
    if X.ndim == 1:
        X = X[:,None]

    dtypes = []
    for feat in X.T:
        dtype = feat.dtype.type
        if not issubclass(dtype, (np.int, np.float)):
            raise ValueError('data must be type int or float, '\
                             'but found type: {}'.format(dtype))

        # if issubclass(dtype, np.int) or (len(np.unique(feat))/len(feat) < ratio):
        if (len(np.unique(feat))/len(feat) < ratio) and \
           ((np.min(feat)) == 0) and (np.max(feat) == len(np.unique(feat)) - 1):
            dtypes.append('categorical')
            continue
        dtypes.append('numerical')

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
    if y.dtype.kind == "O":
        raise ValueError("Targets must be numerical, but found {}".format(y))
    if np.any(np.isnan(link.link(y, dist))):
        raise ValueError('y data is not in domain of {} link function. ' \
                         'Expected domain: {}, but found {}' \
                         .format(link, get_link_domain(link, dist),
                                 [float('%.2f'%np.min(y)),
                                  float('%.2f'%np.max(y))]))
    return y

def make_2d(array):
    """
    tiny tool to expand 1D arrays the way i want
    """
    if array.ndim < 2:
        warnings.warn('Expected 2D input data array, found {}. '\
                      'Expanding to 2D'.format(array.ndim))
        array = np.atleast_1d(array)[:,None]
    return array


def check_X(X, n_feats=None):
    """
    tool to ensure that X is 2 dimensional

    Parameters
    ----------
    X : array-like
    n_feats : int. default: None
              represents number of features that X should have.
              not enforced if n_feats is None.

    Returns
    -------
    X : array with ndims == 2 containing validated X-data
    """
    X = make_2d(X)
    if X.ndim > 2:
        raise ValueError('X must be a matrix or vector. '\
                         'found shape {}'.format(X.shape))
    if n_feats is not None:
        if X.shape[1] != n_feats:
           raise ValueError('X data must have {} features, '\
                            'but found {}'.format(n_feats, X.shape[1]))
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
    if len(X) != len(y):
        raise ValueError('Inconsistent input and output data shapes. '\
                         'found X: {} and y: {}'.format(X.shape, y.shape))

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

    if M >= width:
        raise ValueError('desired width is {}, '\
                         'but max data length is {}'.format(width, M))

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

def gen_edge_knots(data, dtype):
        """
        generate knots from data quantiles

        for discrete data, assumes k categories in [0, k-1] interval
        """
        if dtype not in ['categorical', 'numerical']:
            raise ValueError('unsupported dtype: {}'.format(dtype))
        if dtype == 'categorical':
            return np.r_[np.min(data) - 0.5, np.unique(data) + 0.5]
        else:
            return np.r_[np.min(data), np.max(data)]

def b_spline_basis(x, edge_knots, n_splines=20, spline_order=3, sparse=True):
    """
    tool to generate b-spline basis using vectorized De Boor recursion
    the basis functions extrapolate linearly past the end-knots.

    Parameters
    ----------
    x : array-like, with ndims == 1.
    edge_knots : array-like contaning locations of the 2 edge knots.
    n_splines : int. number of splines to generate. must be >= spline_order+1
                default: 20
    spline_order : int. order of spline basis to create
                   default: 3
    sparse : boolean. whether to return a sparse basis matrix or not.
             default: True

    Returns
    -------
    basis : sparse csc matrix or array containing b-spline basis functions of order: order
            default: sparse csc matrix
    """
    assert np.ravel(x).ndim == 1, 'data must be 1-D, but found {}'.format(np.ravel(x).ndim)
    assert (n_splines >= 1) and (type(n_splines) is int), 'n_splines must be int >= 1'
    assert (spline_order >= 0) and (type(spline_order) is int), 'spline_order must be int >= 1'
    assert n_splines >= spline_order + 1, \
           'n_splines must be >= spline_order + 1. found: n_splines = {} and spline_order = {}'.format(n_splines, spline_order)

    # rescale edge_knots to [0,1], and generate boundary knots
    edge_knots = np.sort(deepcopy(edge_knots))
    offset = edge_knots[0]
    scale = edge_knots[-1] - edge_knots[0]
    boundary_knots = np.linspace(0, 1, 1 + n_splines - spline_order)

    # rescale x as well
    x = (np.ravel(deepcopy(x)) - offset) / scale
    x_extrapolte_left = (x < 0)
    x_extrapolte_right = (x > 1)
    x_interpolate = ~(x_extrapolte_right + x_extrapolte_left)
    x = np.atleast_2d(x).T
    n = len(x)

    # augment knots
    aug_knots = np.r_[boundary_knots.min() * np.ones(spline_order), np.sort(boundary_knots), boundary_knots.max() * np.ones(spline_order)]
    # prepare Haar Basis
    bases = (x >= aug_knots[:-1]).astype(np.int) * (x < aug_knots[1:]).astype(np.int) # haar bases
    try:
        bases[(x >= aug_knots[-1])[:,0], -spline_order-1] = 1 # want the last basis function extend past the boundary
        bases[(x < aug_knots[0])[:,0], spline_order + 1] = 1
    except IndexError as e:
        warnings.warn('Trying to create a feature function with only 1 spline. '\
                      'This is pointless.',
                      stacklevel=2)

    maxi = len(aug_knots) - 1

    # do recursion from Hastie et al.
    for m in range(2, spline_order+2):
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
            grad_left = -1/(boundary_knots[1] - boundary_knots[0]) * (spline_order)
            bases[x_extrapolte_left, :1] = grad_left * x[x_extrapolte_left] + 1
            bases[x_extrapolte_left, 1:2] = -grad_left * x[x_extrapolte_left]
        if any(x_extrapolte_right):
            grad_right = -1/(boundary_knots[-2] - boundary_knots[-1]) * (spline_order)
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
        return [[arg] for arg in args[0]]
