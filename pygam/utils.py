"""
Pygam utilities
"""

from __future__ import division
from copy import deepcopy
import numbers
import sys
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

class OptimizationError(ValueError):
    """Exception class to raise if PIRLS optimization fails
    """


def cholesky(A, sparse=True, verbose=True):
    """
    Choose the best possible cholesky factorizor.

    if possible, import the Scikit-Sparse sparse Cholesky method.
    Permutes the output L to ensure A = L.H . L

    otherwise defaults to numpy's non-sparse version

    Parameters
    ----------
    A : array-like
        array to decompose
    sparse : boolean, default: True
        whether to return a sparse array
    verbose : bool, default: True
        whether to print warnings
    """
    if SKSPIMPORT:
        A = sp.sparse.csc_matrix(A)
        try:
            F = spcholesky(A)

            # permutation matrix P
            P = sp.sparse.lil_matrix(A.shape)
            p = F.P()
            P[np.arange(len(p)), p] = 1

            # permute
            L = F.L()
            L = P.T.dot(L)
        except CholmodNotPositiveDefiniteError as e:
            raise NotPositiveDefiniteError('Matrix is not positive definite')

        if sparse:
            return L.T # upper triangular factorization
        return L.T.A # upper triangular factorization

    else:
        msg = 'Could not import Scikit-Sparse or Suite-Sparse.\n'\
              'This will slow down optimization for models with '\
              'monotonicity/convexity penalties and many splines.\n'\
              'See installation instructions for installing '\
              'Scikit-Sparse and Suite-Sparse via Conda.'
        if verbose:
            warnings.warn(msg)

        if sp.sparse.issparse(A):
            A = A.A

        try:
            L = sp.linalg.cholesky(A, lower=False)
        except LinAlgError as e:
            raise NotPositiveDefiniteError('Matrix is not positive definite')

        if sparse:
            return sp.sparse.csc_matrix(L)
        return L


def make_2d(array, verbose=True):
    """
    tiny tool to expand 1D arrays the way i want

    Parameters
    ----------
    array : array-like

    verbose : bool, default: True
        whether to print warnings

    Returns
    -------
    np.array of with ndim = 2
    """
    array = np.asarray(array)
    if array.ndim < 2:
        msg = 'Expected 2D input data array, but found {}D. '\
              'Expanding to 2D.'.format(array.ndim)
        if verbose:
            warnings.warn(msg)
        array = np.atleast_1d(array)[:,None]
    return array


def check_array(array, force_2d=False, n_feats=None, ndim=None,
                min_samples=1, name='Input data', verbose=True):
    """
    tool to perform basic data validation.
    called by check_X and check_y.

    ensures that data:
    - is ndim dimensional
    - contains float-compatible data-types
    - has at least min_samples
    - has n_feats
    - is finite

    Parameters
    ----------
    array : array-like
    force_2d : boolean, default: False
        whether to force a 2d array. Setting to True forces ndim = 2
    n_feats : int, default: None
              represents number of features that the array should have.
              not enforced if n_feats is None.
    ndim : int default: None
        number of dimensions expected in the array
    min_samples : int, default: 1
    name : str, default: 'Input data'
        name to use when referring to the array
    verbose : bool, default: True
        whether to print warnings

    Returns
    -------
    array : validated array
    """
    # make array
    if force_2d:
        array = make_2d(array, verbose=verbose)
        ndim = 2
    else:
        array = np.array(array)

    # cast to float
    dtype = array.dtype
    if dtype.kind not in ['i', 'f']:
        try:
            array = array.astype('float')
        except ValueError as e:
            raise ValueError('{} must be type int or float, '\
                             'but found type: {}\n'\
                             'Try transforming data with a LabelEncoder first.'\
                             .format(name, dtype.type))

    # check finite
    if not(np.isfinite(array).all()):
        raise ValueError('{} must not contain Inf nor NaN'.format(name))

    # check ndim
    if ndim is not None:
        if array.ndim != ndim:
            raise ValueError('{} must have {} dimensions. '\
                             'found shape {}'.format(name, ndim, array.shape))

    # check n_feats
    if n_feats is not None:
        m = array.shape[1]
        if m != n_feats:
           raise ValueError('{} must have {} features, '\
                            'but found {}'.format(name, n_feats, m))

    # minimum samples
    n = array.shape[0]
    if n < min_samples:
        raise ValueError('{} should have at least {} samples, '\
                         'but found {}'.format(name, min_samples, n))

    return array


def check_y(y, link, dist, min_samples=1, verbose=True):
    """
    tool to ensure that the targets:
    - are in the domain of the link function
    - are numerical
    - have at least min_samples
    - is finite

    Parameters
    ----------
    y : array-like
    link : Link object
    dist : Distribution object
    min_samples : int, default: 1
    verbose : bool, default: True
        whether to print warnings

    Returns
    -------
    y : array containing validated y-data
    """
    y = np.ravel(y)

    y = check_array(y, force_2d=False, min_samples=min_samples, ndim=1,
                    name='y data', verbose=verbose)

    warnings.filterwarnings('ignore', 'divide by zero encountered in log')
    warnings.filterwarnings('ignore', 'invalid value encountered in log')
    if np.any(np.isnan(link.link(y, dist))):
        raise ValueError('y data is not in domain of {} link function. ' \
                         'Expected domain: {}, but found {}' \
                         .format(link, get_link_domain(link, dist),
                                 [float('%.2f'%np.min(y)),
                                  float('%.2f'%np.max(y))]))
    warnings.resetwarnings()

    return y

def check_X(X, n_feats=None, min_samples=1, edge_knots=None, dtypes=None,
            features=None, verbose=True):
    """
    tool to ensure that X:
    - is 2 dimensional
    - contains float-compatible data-types
    - has at least min_samples
    - has n_feats
    - has categorical features in the right range
    - is finite

    Parameters
    ----------
    X : array-like
    n_feats : int. default: None
              represents number of features that X should have.
              not enforced if n_feats is None.
    min_samples : int, default: 1
    edge_knots : list of arrays, default: None
    dtypes : list of strings, default: None
    features : list of ints,
        which features are considered by the model
    verbose : bool, default: True
        whether to print warnings

    Returns
    -------
    X : array with ndims == 2 containing validated X-data
    """
    # check all features are there
    if bool(features):
        features = flatten(features)
        max_feat = max(flatten(features))

        if n_feats is None:
            n_feats = max_feat

        n_feats = max(n_feats, max_feat)

    # basic diagnostics
    X = check_array(X, force_2d=True, n_feats=n_feats, min_samples=min_samples,
                    name='X data', verbose=verbose)

    # check our categorical data has no new categories
    if (edge_knots is not None) and (dtypes is not None) and (features is not None):

        # get a flattened list of tuples
        edge_knots = flatten(edge_knots)[::-1]
        dtypes = flatten(dtypes)
        assert len(edge_knots) % 2 == 0 # sanity check

        # form pairs
        n = len(edge_knots) // 2
        edge_knots = [(edge_knots.pop(), edge_knots.pop()) for _ in range(n)]

        # check each categorical term
        for i, ek in enumerate(edge_knots):
            dt = dtypes[i]
            feature = features[i]
            x = X[:, feature]

            if dt == 'categorical':
                min_ = ek[0]
                max_ = ek[-1]
                if (np.unique(x) < min_).any() or \
                   (np.unique(x) > max_).any():
                    min_ += .5
                    max_ -= 0.5
                    raise ValueError('X data is out of domain for categorical '\
                                     'feature {}. Expected data on [{}, {}], '\
                                     'but found data on [{}, {}]'\
                                     .format(i, min_, max_, x.min(), x.max()))

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


def check_param(param, param_name, dtype, constraint=None, iterable=True,
                max_depth=2):
    """
    checks the dtype of a parameter,
    and whether it satisfies a numerical contraint

    Parameters
    ---------
    param : object
    param_name : str, name of the parameter
    dtype : str, desired dtype of the parameter
    contraint : str, default: None
                numerical constraint of the parameter.
                if None, no constraint is enforced
    iterable : bool, default: True
               whether to allow iterable param
    max_depth : int, default: 2
                maximum nesting of the iterable.
                only used if iterable == True
    Returns
    -------
    list of validated and converted parameter(s)
    """
    msg = []
    msg.append(param_name + " must be "+ dtype)
    if iterable:
        msg.append(" or nested iterable of depth " + str(max_depth) +
                   " containing " + dtype + "s")

    msg.append(", but found " + param_name + " = {}".format(repr(param)))

    if constraint is not None:
        msg = (" " + constraint).join(msg)
    else:
        msg = ''.join(msg)

    # check param is numerical
    try:
        param_dt = np.array(flatten(param))# + np.zeros_like(flatten(param), dtype='int')
        # param_dt = np.array(param).astype(dtype)
    except (ValueError, TypeError):
        raise TypeError(msg)

    # check iterable
    if iterable:
        if check_iterable_depth(param) > max_depth:
            raise TypeError(msg)
    if (not iterable) and isiterable(param):
        raise TypeError(msg)

    # check param is correct dtype
    if not (param_dt == np.array(flatten(param)).astype(float)).all():
        raise TypeError(msg)

    # check constraint
    if constraint is not None:
        if not (eval('np.' + repr(param_dt) + constraint)).all():
            raise ValueError(msg)

    return param

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


def load_diagonal(cov, load=None):
        """Return the given square matrix with a small amount added to the diagonal
        to make it positive semi-definite.
        """
        n, m = cov.shape
        assert n == m, "matrix must be square, but found shape {}".format((n, m))

        if load is None:
            load = np.sqrt(np.finfo(np.float64).eps) # machine epsilon
        return cov + np.eye(n) * load


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


# Credit to Hugh Bothwell from http://stackoverflow.com/questions/5084743/how-to-print-pretty-string-output-in-python
class TablePrinter(object):
    "Print a list of dicts as a table"
    def __init__(self, fmt, sep=' ', ul=None):
        """
        @param fmt: list of tuple(heading, key, width)
                        heading: str, column label
                        key: dictionary key to value to print
                        width: int, column width in chars
        @param sep: string, separation between columns
        @param ul: string, character to underline column label, or None for no underlining
        """
        super(TablePrinter,self).__init__()
        self.fmt   = str(sep).join('{lb}{0}:{1}{rb}'.format(key, width, lb='{', rb='}') for heading,key,width in fmt)
        self.head  = {key:heading for heading,key,width in fmt}
        self.ul    = {key:str(ul)*width for heading,key,width in fmt} if ul else None
        self.width = {key:width for heading,key,width in fmt}

    def row(self, data):
        if sys.version_info < (3,):
            return self.fmt.format(**{ k:str(data.get(k,''))[:w] for k,w in self.width.iteritems() })
        else:
            return self.fmt.format(**{ k:str(data.get(k,''))[:w] for k,w in self.width.items() })

    def __call__(self, dataList):
        _r = self.row
        res = [_r(data) for data in dataList]
        res.insert(0, _r(self.head))
        if self.ul:
            res.insert(1, _r(self.ul))
        return '\n'.join(res)


def space_row(left, right, filler=' ', total_width=-1):
    """space the data in a row with optional filling

    Arguments
    ---------
    left : str, to be aligned left
    right : str, to be aligned right
    filler : str, default ' '.
        must be of length 1
    total_width : int, width of line.
        if negative number is specified,
        then that number of spaces is used between the left and right text

    Returns
    -------
    str
    """
    left = str(left)
    right = str(right)
    filler = str(filler)[:1]

    if total_width < 0:
        spacing = - total_width
    else:
        spacing = total_width - len(left) - len(right)

    return left + filler * spacing + right

def sig_code(p_value):
    """create a significance code in the style of R's lm

    Arguments
    ---------
    p_value : float on [0, 1]

    Returns
    -------
    str
    """
    assert 0 <= p_value <= 1, 'p_value must be on [0, 1]'
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    if p_value < 0.1:
        return '.'
    return ' '

def gen_edge_knots(data, dtype, verbose=True):
    """
    generate uniform knots from data including the edges of the data

    for discrete data, assumes k categories in [0, k-1] interval

    Parameters
    ----------
    data : array-like with one dimension
    dtype : str in {'categorical', 'numerical'}
    verbose : bool, default: True
        whether to print warnings

    Returns
    -------
    np.array containing ordered knots
    """
    if dtype not in ['categorical', 'numerical']:
        raise ValueError('unsupported dtype: {}'.format(dtype))
    if dtype == 'categorical':
        return np.r_[np.min(data) - 0.5, np.max(data) + 0.5]
    else:
        knots = np.r_[np.min(data), np.max(data)]
        if knots[0] == knots[1] and verbose:
            warnings.warn('Data contains constant feature. '\
                          'Consider removing and setting fit_intercept=True',
                          stacklevel=2)
        return knots

def b_spline_basis(x, edge_knots, n_splines=20, spline_order=3, sparse=True,
                   verbose=True):
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
    verbose : bool, default: True
        whether to print warnings

    Returns
    -------
    basis : sparse csc matrix or array containing b-spline basis functions
            with shape (len(x), n_splines)
    """
    if np.ravel(x).ndim != 1:
        raise ValueError('Data must be 1-D, but found {}'\
                         .format(np.ravel(x).ndim))

    if (n_splines < 1) or not isinstance(n_splines, numbers.Integral):
        raise ValueError('n_splines must be int >= 1')

    if (spline_order < 0) or not isinstance(spline_order, numbers.Integral):
        raise ValueError('spline_order must be int >= 1')

    if n_splines < spline_order + 1:
        raise ValueError('n_splines must be >= spline_order + 1. '\
                         'found: n_splines = {} and spline_order = {}'\
                         .format(n_splines, spline_order))

    if n_splines == 0 and verbose:
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

        # left sub-basis
        num = (x - aug_knots[:maxi])
        num *= bases[:, :maxi]
        denom = aug_knots[m-1 : maxi+m-1] - aug_knots[:maxi]
        left = num/denom

        # right sub-basis
        num = (aug_knots[m : maxi+m] - x) * bases[:, 1:maxi+1]
        denom = aug_knots[m:maxi+m] - aug_knots[1 : maxi+1]
        right = num/denom

        # track previous bases and update
        prev_bases = bases[-2:]
        bases = left + right

    # extrapolate
    # since we have repeated end-knots, only the last 2 basis functions are
    # non-zero at the end-knots, and they have equal and opposite gradient.
    if (any(x_extrapolte_r) or any(x_extrapolte_l)) and spline_order>0:
        bases[~x_interpolate] = 0.

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
    """convenience tool to detect if something is iterable.
    in python3, strings count as iterables to we have the option to exclude them

    Parameters:
    -----------
    obj : object to analyse
    reject_string : bool, whether to ignore strings

    Returns:
    --------
    bool, if the object is itereable.
    """

    iterable =  hasattr(obj, '__len__')

    if reject_string:
        iterable = iterable and not isinstance(obj, str)

    return iterable

def check_iterable_depth(obj, max_depth=100):
    """find the maximum depth of nesting of the iterable

    Parameters
    ----------
    obj : iterable
    max_depth : int, default: 100
        maximum depth beyond which we stop counting

    Returns
    -------
    int
    """
    def find_iterables(obj):
        iterables = []
        for item in obj:
            if isiterable(item):
                iterables += list(item)
        return iterables

    depth = 0
    while (depth < max_depth) and isiterable(obj) and len(obj) > 0:
        depth += 1
        obj = find_iterables(obj)
    return depth

def flatten(iterable):
    """convenience tool to flatten any nested iterable

    example:

        flatten([[[],[4]],[[[5,[6,7, []]]]]])
        >>> [4, 5, 6, 7]

        flatten('hello')
        >>> 'hello'

    Parameters
    ----------
    iterable

    Returns
    -------
    flattened object
    """
    if isiterable(iterable):
        flat = []
        for item in list(iterable):
            item = flatten(item)
            if not isiterable(item):
                item = [item]
            flat += item
        return flat
    else:
        return iterable


def tensor_product(a, b, reshape=True):
    """
    compute the tensor protuct of two matrices a and b

    if a is (n, m_a), b is (n, m_b),
    then the result is
        (n, m_a * m_b) if reshape = True.
    or
        (n, m_a, m_b) otherwise

    Parameters
    ---------
    a : array-like of shape (n, m_a)

    b : array-like of shape (n, m_b)

    reshape : bool, default True
        whether to reshape the result to be 2-dimensional ie
        (n, m_a * m_b)
        or return a 3-dimensional tensor ie
        (n, m_a, m_b)

    Returns
    -------
    dense np.ndarray of shape
        (n, m_a * m_b) if reshape = True.
    or
        (n, m_a, m_b) otherwise
    """
    assert a.ndim == 2, 'matrix a must be 2-dimensional, but found {} dimensions'.format(a.ndim)
    assert b.ndim == 2, 'matrix b must be 2-dimensional, but found {} dimensions'.format(b.ndim)

    na, ma = a.shape
    nb, mb = b.shape

    if na != nb:
        raise ValueError('both arguments must have the same number of samples')

    if sp.sparse.issparse(a):
        a = a.A

    if sp.sparse.issparse(b):
        b = b.A

    tensor = a[..., :, None] * b[..., None, :]

    if reshape:
        return tensor.reshape(na, ma * mb)

    return tensor
