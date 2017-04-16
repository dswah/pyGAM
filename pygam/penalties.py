"""
Penalty matrix generators
"""

import scipy as sp
import numpy as np


def cont_P(n, diff_order=1):
    """
    Builds a penalty matrix for P-Splines with continuous features.
    Penalizes the squared differences between adjacent basis coefficients.

    Parameters
    ----------
    n : int
        number of splines

    diff_order: int, default: 1
        differencing order.
        when the order is 1, we penalize second order derivatives,
        when the order is 2, we penalize third order derivatives, etc

    Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    if n==1:
        # no second order derivative for constant functions
        return sp.sparse.csc_matrix(0.)
    D = sparse_diff(sp.sparse.identity(n).tocsc(), n=diff_order)
    return D.dot(D.T)

def cat_P(n):
    """
    Builds a penalty matrix for P-Splines with categorical features.
    Penalizes the squared value of each basis coefficient.

    Parameters
    ----------
    n : int
        number of splines

    Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    return sp.sparse.eye(n).tocsc()

def wrap_penalty(p, fit_linear):
    """
    tool to account for unity penalty on the linear term of any feature.

    Parameters
    ----------
    p : callable.
      penalty-matrix-generating function.
    fit_linear : boolean.
      whether the current feature has a linear term or not.

    Returns
    -------
    wrapped_p : callable
      modified penalty-matrix-generating function
    """
    def wrapped_p(n):
        if fit_linear:
            if n == 1:
                return sp.sparse.block_diag([1.], format='csc')
            return sp.sparse.block_diag([1., p(n-1)], format='csc')
        else:
            return p(n)
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
