"""
Penalty matrix generators
"""

import scipy as sp
import numpy as np


def cont_P(n, diff_order=1):
    """
    builds a default proto-penalty matrix for P-Splines for continuous features.
    penalizes the squared differences between adjacent basis coefficients.
    """
    if n==1:
        return sp.sparse.csc_matrix(0.) # no second order derivative for constant functions
    D = np.diff(np.eye(n), n=diff_order)
    return sp.sparse.csc_matrix(D.dot(D.T))

def cat_P(n):
    """
    builds a default proto-penalty matrix for P-Splines for categorical features.
    penalizes the squared value of each basis coefficient.
    """
    return sp.sparse.csc_matrix(np.eye(n))

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
            return sp.sparse.block_diag([1., p(n-1)], format='csc')
        else:
            return p(n)
    return wrapped_p
