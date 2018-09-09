"""
Penalty matrix generators
"""

import scipy as sp
import numpy as np


def derivative(n, coef, derivative=2):
    """
    Builds a penalty matrix for P-Splines with continuous features.
    Penalizes the squared differences between basis coefficients.

    Parameters
    ----------
    n : int
        number of splines

    coef : unused
        for compatibility with constraints

    derivative: int, default: 2
        which derivative do we penalize.
        derivative is 1, we penalize 1st order derivatives,
        derivative is 2, we penalize 2nd order derivatives, etc

    Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    if n == 1:
        # no derivative for constant functions
        return sp.sparse.csc_matrix(0.)
    D = sparse_diff(sp.sparse.identity(n).tocsc(), n=derivative)
    return D.dot(D.T).tocsc()

def l2(n, coef):
    """
    Builds a penalty matrix for P-Splines with categorical features.
    Penalizes the squared value of each basis coefficient.

    Parameters
    ----------
    n : int
        number of splines

    coef : unused
        for compatibility with constraints

    Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    return sp.sparse.eye(n).tocsc()

def monotonicity_(n, coef, increasing=True):
    """
    Builds a penalty matrix for P-Splines with continuous features.
    Penalizes violation of monotonicity in the feature function.

    Parameters
    ----------
    n : int
        number of splines
    coef : array-like
        coefficients of the feature function
    increasing : bool, default: True
        whether to enforce monotic increasing, or decreasing functions
    Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    if n != len(coef.ravel()):
        raise ValueError('dimension mismatch: expected n equals len(coef), '\
                         'but found n = {}, coef.shape = {}.'\
                         .format(n, coef.shape))

    if n==1:
        # no monotonic penalty for constant functions
        return sp.sparse.csc_matrix(0.)

    if increasing:
        # only penalize the case where coef_i-1 > coef_i
        mask = sp.sparse.diags((np.diff(coef.ravel()) < 0).astype(float))
    else:
        # only penalize the case where coef_i-1 < coef_i
        mask = sp.sparse.diags((np.diff(coef.ravel()) > 0).astype(float))

    derivative = 1
    D = sparse_diff(sp.sparse.identity(n).tocsc(), n=derivative) * mask
    return D.dot(D.T).tocsc()

def monotonic_inc(n, coef):
    """
    Builds a penalty matrix for P-Splines with continuous features.
    Penalizes violation of a monotonic increasing feature function.

    Parameters
    ----------
    n : int
        number of splines
    coef : array-like, coefficients of the feature function

    Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    return monotonicity_(n, coef, increasing=True)

def monotonic_dec(n, coef):
    """
    Builds a penalty matrix for P-Splines with continuous features.
    Penalizes violation of a monotonic decreasing feature function.

    Parameters
    ----------
    n : int
        number of splines
    coef : array-like
        coefficients of the feature function

    Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    return monotonicity_(n, coef, increasing=False)

def convexity_(n, coef, convex=True):
    """
    Builds a penalty matrix for P-Splines with continuous features.
    Penalizes violation of convexity in the feature function.

    Parameters
    ----------
    n : int
        number of splines
    coef : array-like
        coefficients of the feature function
    convex : bool, default: True
        whether to enforce convex, or concave functions
    Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    if n != len(coef.ravel()):
        raise ValueError('dimension mismatch: expected n equals len(coef), '\
                         'but found n = {}, coef.shape = {}.'\
                         .format(n, coef.shape))

    if n==1:
        # no convex penalty for constant functions
        return sp.sparse.csc_matrix(0.)

    if convex:
        mask = sp.sparse.diags((np.diff(coef.ravel(), n=2) < 0).astype(float))
    else:
        mask = sp.sparse.diags((np.diff(coef.ravel(), n=2) > 0).astype(float))

    derivative = 2
    D = sparse_diff(sp.sparse.identity(n).tocsc(), n=derivative) * mask
    return D.dot(D.T).tocsc()

def convex(n, coef):
    """
    Builds a penalty matrix for P-Splines with continuous features.
    Penalizes violation of a convex feature function.

    Parameters
    ----------
    n : int
        number of splines
    coef : array-like
        coefficients of the feature function

    Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    return convexity_(n, coef, convex=True)

def concave(n, coef):
    """
    Builds a penalty matrix for P-Splines with continuous features.
    Penalizes violation of a concave feature function.

    Parameters
    ----------
    n : int
        number of splines
    coef : array-like
        coefficients of the feature function

    Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    return convexity_(n, coef, convex=False)

# def circular(n, coef):
#     """
#     Builds a penalty matrix for P-Splines with continuous features.
#     Penalizes violation of a circular feature function.
#
#     Parameters
#     ----------
#     n : int
#         number of splines
#     coef : unused
#         for compatibility with constraints
#
#     Returns
#     -------
#     penalty matrix : sparse csc matrix of shape (n,n)
#     """
#     if n != len(coef.ravel()):
#         raise ValueError('dimension mismatch: expected n equals len(coef), '\
#                          'but found n = {}, coef.shape = {}.'\
#                          .format(n, coef.shape))
#
#     if n==1:
#         # no first circular penalty for constant functions
#         return sp.sparse.csc_matrix(0.)
#
#     row = np.zeros(n)
#     row[0] = 1
#     row[-1] = -1
#     P = sp.sparse.vstack([row, sp.sparse.csc_matrix((n-2, n)), row[::-1]])
#     return P.tocsc()

def none(n, coef):
    """
    Build a matrix of zeros for features that should go unpenalized

    Parameters
    ----------
    n : int
        number of splines
    coef : unused
        for compatibility with constraints

    Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    return sp.sparse.csc_matrix(np.zeros((n, n)))

def wrap_penalty(p, fit_linear, linear_penalty=0.):
    """
    tool to account for unity penalty on the linear term of any feature.

    example:
        p = wrap_penalty(derivative, fit_linear=True)(n, coef)

    Parameters
    ----------
    p : callable.
        penalty-matrix-generating function.
    fit_linear : boolean.
        whether the current feature has a linear term or not.
    linear_penalty : float, default: 0.
        penalty on the linear term

    Returns
    -------
    wrapped_p : callable
      modified penalty-matrix-generating function
    """
    def wrapped_p(n, *args):
        if fit_linear:
            if n == 1:
                return sp.sparse.block_diag([linear_penalty], format='csc')
            return sp.sparse.block_diag([linear_penalty,
                                         p(n-1, *args)], format='csc')
        else:
            return p(n, *args)
    return wrapped_p

def sparse_diff(array, n=1, axis=-1):
    """
    A ported sparse version of np.diff.
    Uses recursion to compute higher order differences

    Parameters
    ----------
    array : sparse array
    n : int, default: 1
        differencing order
    axis : int, default: -1
        axis along which differences are computed

    Returns
    -------
    diff_array : sparse array
                 same shape as input array,
                 but 'axis' dimension is smaller by 'n'.
    """
    if (n < 0) or (int(n) != n):
        raise ValueError('Expected order is non-negative integer, '\
                         'but found: {}'.format(n))
    if not sp.sparse.issparse(array):
        warnings.warn('Array is not sparse. Consider using numpy.diff')

    if n == 0:
        return array

    nd = array.ndim
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)

    A = sparse_diff(array, n-1, axis=axis)
    return A[slice1] - A[slice2]


PENALTIES = {'auto': 'auto',
             'derivative': derivative,
             'l2': l2,
             'none': none,
            }

CONSTRAINTS = {'convex': convex,
               'concave': concave,
               'monotonic_inc': monotonic_inc,
               'monotonic_dec': monotonic_dec,
               'none': none
              }
