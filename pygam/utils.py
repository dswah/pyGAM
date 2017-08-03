"""
Pygam utilities
"""

from __future__ import division
from copy import deepcopy
import warnings

import scipy as sp
from scipy import sparse
import numpy as np
from numpy.linalg import LinAlgError

try:
  from sksparse.cholmod import cholesky as spcholesky
  from sksparse.test_cholmod import CholmodNotPositiveDefiniteError
  SKSPIMPORT = True
except ImportError:
  SKSPIMPORT = False


class NotPositiveDefiniteError(ValueError):
    """Exception class to raise if a matrix is not positive definite
    """


def cholesky(A, sparse=True):
    """
    Choose the best possible cholesky factorizor.

    if possible, import the Scikit-Sparse sparse Cholesky method.
    Permutes the output L to ensure A = L . L.H

    otherwise defaults to numpy's non-sparse version

    Parameters
    ----------
    A : array-like
        array to decompose
    sparse : boolean, default: True
        whether to return a sparse array
    """
    if SKSPIMPORT:
        A = sp.sparse.csc_matrix(A)
        F = spcholesky(A)

        # permutation matrix P
        P = sp.sparse.lil_matrix(A.shape)
        p = F.P()
        P[np.arange(len(p)), p] = 1

        # permute
        try:
            L = F.L()
            L = P.T.dot(L)
        except CholmodNotPositiveDefiniteError as e:
            raise NotPositiveDefiniteError('Matrix is not positive definite')

        if sparse:
            return L
        return L.todense()

    else:
        msg = 'Could not import Scikit-Sparse or Suite-Sparse.\n'\
              'This will slow down optimization for models with '\
              'monotonicity/convexity penalties and many splines.\n'\
              'See installation instructions for installing '\
              'Scikit-Sparse and Suite-Sparse via Conda.'
        warnings.warn(msg)

        if sp.sparse.issparse(A):
            A = A.todense()

        try:
            L = np.linalg.cholesky(A)
        except LinAlgError as e:
            raise NotPositiveDefiniteError('Matrix is not positive definite')

        if sparse:
            return sp.sparse.csc_matrix(L)
        return L


def generate_X_grid(gam, n=500):
    """
    tool to create a nice grid of X data if no X data is supplied

    array is sorted by feature and uniformly spaced, so the marginal and joint
    distributions are likely wrong

    Parameters
    ----------
    gam : GAM instance
    n : int, default: 500
        number of data points to create

    Returns
    -------
    np.array of shape (n, n_features)
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
        dtype = feat.dtype.kind
        if dtype not in ['f', 'i']:
            raise ValueError('Data must be type int or float, '\
                             'but found type: {}'.format(feat.dtype))

        if dtype == 'f':
            if not(np.isfinite(feat).all()):
                raise ValueError('Data must not contain Inf nor NaN')

        # if issubclass(dtype, np.int) or \
        # (len(np.unique(feat))/len(feat) < ratio):
        if (len(np.unique(feat))/len(feat) < ratio) and \
           ((np.min(feat)) == 0) and (np.max(feat) == len(np.unique(feat)) - 1):
            dtypes.append('categorical')
            continue
        dtypes.append('numerical')

    return dtypes


def check_y(y, link, dist, min_samples=1):
    """
    tool to ensure that the targets:
    - are in the domain of the link function
    - are numerical
    - have at least min_samples

    Parameters
    ----------
    y : array-like
    link : Link object
    dist : Distribution object
    min_samples : int, default: 1

    Returns
    -------
    y : array containing validated y-data
    """
    y = np.ravel(y)
    if y.dtype.kind not in['f', 'i']:
        try:
            y = y.astype('float')
        except ValueError as e:
            raise ValueError("Targets must be type int or float, "\
                             "but found {}".format(y))

    warnings.filterwarnings('ignore', 'divide by zero encountered in log')
    if np.any(np.isnan(link.link(y, dist))):
        raise ValueError('y data is not in domain of {} link function. ' \
                         'Expected domain: {}, but found {}' \
                         .format(link, get_link_domain(link, dist),
                                 [float('%.2f'%np.min(y)),
                                  float('%.2f'%np.max(y))]))
    warnings.resetwarnings()

    if len(y) < min_samples:
        raise ValueError('targets should have at least {} samples, '\
                         'but found {}'.format(min_samples, len(y)))

    if not(np.isfinite(y).all()):
        raise ValueError('y data must not contain Inf nor NaN')

    return y

def make_2d(array):
    """
    tiny tool to expand 1D arrays the way i want

    Parameters
    ----------
    array : array-like

    Returns
    -------
    np.array of with ndim = 2
    """
    array = np.asarray(array)
    if array.ndim < 2:
        msg = 'Expected 2D input data array, but found {}D. '\
              'Expanding to 2D.'.format(array.ndim)
        warnings.warn(msg)
        array = np.atleast_1d(array)[:,None]
    return array


def check_X(X, n_feats=None, min_samples=1, edge_knots=None, dtypes=None):
    """
    tool to ensure that X:
    - is 2 dimensional
    - contains float-compatible data-types
    - has at least min_samples
    - has n_feats
    - has caegorical features in the right range

    Parameters
    ----------
    X : array-like
    n_feats : int. default: None
              represents number of features that X should have.
              not enforced if n_feats is None.
    min_samples : int, default: 1
    edge_knots : list of arrays, default: None
    dtypes : list of strings, default: None

    Returns
    -------
    X : array with ndims == 2 containing validated X-data
    """
    X = make_2d(X)
    if X.ndim > 2:
        raise ValueError('X must be a matrix or vector. '\
                         'found shape {}'.format(X.shape))
    n, m = X.shape
    if n_feats is not None:
        if m != n_feats:
           raise ValueError('X data must have {} features, '\
                            'but found {}'.format(n_feats, m))

    dtype = X.dtype
    if dtype.kind not in ['i', 'f']:
        try:
            X = X.astype('float')
        except ValueError as e:
            raise ValueError('X data must be type int or float, '\
                             'but found type: {}\n'\
                             'Try transforming data with a LabelEncoder first.'\
                             .format(dtype.type))
    if n < min_samples:
        raise ValueError('X data should have at least {} samples, '\
                         'but found {}'.format(min_samples, n))

    if (edge_knots is not None) and (dtypes is not None):
        for i, (dt, ek, feat) in enumerate(zip(dtypes, edge_knots, X.T)):
            if dt == 'categorical':
                min_ = ek[0]
                max_ = ek[-1]
                if (np.unique(feat) < min_).any() or \
                   (np.unique(feat) > max_).any():
                    min_ += .5
                    max_ -= 0.5
                    feat_min = feat.min()
                    feat_max = feat.max()
                    raise ValueError('X data is out of domain for categorical '\
                                     'feature {}. Expected data in [{}, {}], '\
                                     'but found data in [{}, {}]'\
                                     .format(i, min_, max_, feat_min, feat_max))


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

def check_lengths(*arrays):
    """
    tool to ensure input and output data have the same number of samples

    Parameters
    ----------
    *arrays : iterable of arrays to be checked

    Returns
    -------
    None
    """
    lengths = [len(array) for array in arrays]
    if len(np.unique(lengths)) > 1:
        raise ValueError('Inconsistent data lengths: {}'.format(lengths))


def check_param(param, param_name, dtype, iterable=True, constraint=None):
    """
    checks the dtype of a parameter,
    and whether it satisfies a numerical contraint

    Parameters
    ---------
    param : object
    param_name : str, name of the parameter
    dtype : str, desired dtype of the parameter
    iterable : bool, default: True
               whether to allow iterable param
    contraint : str, default: None
                numerical constraint of the parameter.
                if None, no constraint is enforced

    Returns
    -------
    list of validated and converted parameter(s)
    """
    msg = []
    msg.append(param_name + " must be "+ dtype)
    if iterable:
        msg.append(" or iterable of " + dtype + "s")
    msg.append(", but found " + param_name + " = {}".format(repr(param)))

    if constraint is not None:
        msg = (" " + constraint).join(msg)
    else:
        msg = ''.join(msg)

    # check param is numerical
    try:
        param_dt = np.array(param).astype(dtype)
    except ValueError:
        raise ValueError(msg)

    # check iterable
    if (not iterable) and (param_dt.size != 1):
        raise ValueError(msg)

    # check param is correct dtype
    if not (param_dt == np.array(param).astype(float)).all():
        raise ValueError(msg)

    # check constraint
    if constraint is not None:
        if not (eval('np.' + repr(param_dt) + constraint)).all():
            raise ValueError(msg)

    return param_dt.tolist()

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
    out = ((np.atleast_1d(array) * 10**n).round().astype('int') / (10.**n))
    return out.reshape(shape)


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
    keys = np.array(list(data_dict.keys()), dtype='str')
    values = np.array(list(data_dict.values()))
    values = round_to_n_decimal_places(values).astype('str')
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
    generate uniform knots from data including the edges of the data

    for discrete data, assumes k categories in [0, k-1] interval

    Parameters
    ----------
    data : array-like with one dimension
    dtype : str in {'categorical', 'numerical'}

    Returns
    -------
    np.array containing ordered knots
    """
    if dtype not in ['categorical', 'numerical']:
        raise ValueError('unsupported dtype: {}'.format(dtype))
    if dtype == 'categorical':
        return np.r_[np.min(data) - 0.5, np.unique(data) + 0.5]
    else:
        knots = np.r_[np.min(data), np.max(data)]
        if knots[0] == knots[1]:
            warnings.warn('Data contains constant feature. '\
                          'Consider removing and setting fit_intercept=True',
                          stacklevel=2)
        return knots

def b_spline_basis(x, edge_knots, n_splines=20,
                    spline_order=3, sparse=True,
                    clamped=False):
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
    clamped : boolean, default: False
              whether to force repeated knots at the ends of the domain.

              NOTE: when Flase this results in interpretable basis functions
              where creating a linearly incrasing function ammounts to
              assigning linearly increasing coefficients.

              when clamped, this is no longer true and constraints that depend
              on this property, like monotonicity and convexity are no longer
              valid.

    Returns
    -------
    basis : sparse csc matrix or array containing b-spline basis functions
            with shape (len(x), n_splines)
    """
    if np.ravel(x).ndim != 1:
        raise ValueError('Data must be 1-D, but found {}'\
                         .format(np.ravel(x).ndim))

    if (n_splines < 1) or (type(n_splines) is not int):
        raise ValueError('n_splines must be int >= 1')

    if (spline_order < 0) or (type(spline_order) is not int):
        raise ValueError('spline_order must be int >= 1')

    if n_splines < spline_order + 1:
        raise ValueError('n_splines must be >= spline_order + 1. '\
                         'found: n_splines = {} and spline_order = {}'\
                         .format(n_splines, spline_order))

    if n_splines == 0:
        warnings.warn('Requested 1 spline. This is equivalent to '\
                      'fitting an intercept', stacklevel=2)

    # rescale edge_knots to [0,1], and generate boundary knots
    edge_knots = np.sort(deepcopy(edge_knots))
    offset = edge_knots[0]
    scale = edge_knots[-1] - edge_knots[0]
    if scale == 0:
        scale = 1
    boundary_knots = np.linspace(0, 1, 1 + n_splines - spline_order)
    diff = np.diff(boundary_knots[:2])[0]

    # rescale x as well
    x = (np.ravel(deepcopy(x)) - offset) / scale

    # append 0 and 1 in order to get derivatives for extrapolation
    x = np.r_[x, 0., 1.]

    # determine extrapolation indices
    x_extrapolte_l = (x < 0)
    x_extrapolte_r = (x > 1)
    x_interpolate = ~(x_extrapolte_r + x_extrapolte_l)

    # formatting
    x = np.atleast_2d(x).T
    n = len(x)

    # augment knots
    if clamped:
        aug_knots = np.r_[np.zeros(spline_order),
                          boundary_knots,
                          np.ones(spline_order)]
    else:
        aug = np.arange(1, spline_order + 1) * diff
        aug_knots = np.r_[-aug[::-1],
                          boundary_knots,
                          1 + aug]
    aug_knots[-1] += 1e-9 # want last knot inclusive

    # prepare Haar Basis
    bases = (x >= aug_knots[:-1]).astype(np.int) * \
            (x < aug_knots[1:]).astype(np.int)
    bases[-1] = bases[-2][::-1] # force symmetric bases at 0 and 1

    # do recursion from Hastie et al. vectorized
    maxi = len(aug_knots) - 1
    for m in range(2, spline_order + 2):
        maxi -= 1

        # bookkeeping to avoid div by 0
        mask_l = aug_knots[m - 1 : maxi + m - 1] != aug_knots[:maxi]
        mask_r = aug_knots[m : maxi + m] != aug_knots[1 : maxi + 1]

        # left sub-basis
        num = (x - aug_knots[:maxi][mask_l]) * bases[:, :maxi][:, mask_l]
        denom = aug_knots[m-1 : maxi+m-1][mask_l] - aug_knots[:maxi][mask_l]
        left = np.zeros((n, maxi))
        left[:, mask_l] = num/denom

        # right sub-basis
        num = (aug_knots[m : maxi+m][mask_r]-x) * bases[:, 1:maxi+1][:, mask_r]
        denom = aug_knots[m:maxi+m][mask_r] - aug_knots[1 : maxi+1][mask_r]
        right = np.zeros((n, maxi))
        right[:, mask_r] = num/denom

        # track previous bases and update
        prev_bases = bases[-2:]
        bases = left + right

    # extrapolate
    # since we have repeated end-knots, only the last 2 basis functions are
    # non-zero at the end-knots, and they have equal and opposite gradient.
    if (any(x_extrapolte_r) or any(x_extrapolte_l)) and spline_order>0:
        bases[~x_interpolate] = 0.
        if not clamped:
            denom = (aug_knots[spline_order:-1] - aug_knots[: -spline_order - 1])
            left = prev_bases[:, :-1] / denom

            denom = (aug_knots[spline_order+1:] - aug_knots[1: -spline_order])
            right = prev_bases[:, 1:] / denom

            grads = (spline_order) * (left - right)

            if any(x_extrapolte_l):
                val = grads[0] * x[x_extrapolte_l] + bases[-2]
                bases[x_extrapolte_l] = val
            if any(x_extrapolte_r):
                val = grads[1] * (x[x_extrapolte_r] - 1) + bases[-1]
                bases[x_extrapolte_r] = val
        else:
            grad = -spline_order/diff
            if any(x_extrapolte_l):
                bases[x_extrapolte_l, :1] = grad * x[x_extrapolte_l] + 1
                bases[x_extrapolte_l, 1:2] = -grad * x[x_extrapolte_l]

            if any(x_extrapolte_r):
                bases[x_extrapolte_r, -1:] = -grad * (x[x_extrapolte_r] - 1) + 1
                bases[x_extrapolte_r, -2:-1] = grad * (x[x_extrapolte_r] - 1)

    # get rid of the added values at 0, and 1
    bases = bases[:-2]

    if sparse:
        return sp.sparse.csc_matrix(bases)

    return bases


def ylogydu(y, u):
    """
    tool to give desired output for the limit as y -> 0, which is 0

    Parameters
    ----------
    y : array-like of len(n)
    u : array-like of len(n)

    Returns
    -------
    np.array len(n)
    """
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

def isiterable(obj, reject_string=True):
    """
    convenience tool to detect if something is iterable.
    in python3, strings count as iterables to we have the option to exclude them

    Parameters:
    -----------
    obj : object to analyse
    reject_string : bool, whether to ignore strings

    Returns:
    --------
    bool, if the object is itereable.
    """

    iterable =  hasattr(obj, '__iter__')

    if reject_string:
        iterable *= not isinstance(obj, str)

    return iterable
