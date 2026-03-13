"""
Utilities for smoothness selection via outer-loop optimization.

These helpers keep the implementation modular and allow derivative-based or
finite-difference-based optimization around the existing PIRLS routine.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg


def estimate_rank(R, tol_cond=1e12):
    """
    Estimate effective rank by scanning leading principal minors of R.

    Parameters
    ----------
    R : np.ndarray, upper triangular from QR
    tol_cond : float
        maximum allowed condition number
    """
    n = R.shape[0]
    for r in range(n, 0, -1):
        sub = R[:r, :r]
        if np.linalg.cond(sub) < tol_cond:
            return r
    return 1


def qr_decomp(WX, E, tol_cond=1e12):
    """
    Pivoted QR decomposition of [WX; E] per Wood (2008) Sec 3.2.

    Parameters
    ----------
    WX : array (n, q) weighted model matrix
    E : array (p, q) such that E.T @ E = S
    tol_cond : float, condition number threshold for rank detection
    """
    aug = np.vstack([WX, E])
    n, q = WX.shape
    Q, R, piv = scipy.linalg.qr(aug, pivoting=True, mode="economic")
    r = estimate_rank(R, tol_cond=tol_cond)
    Qr = Q[:, :r]
    Rr = R[:r, :r]
    K = Qr[:n, :]
    Rinv = scipy.linalg.solve_triangular(Rr, np.eye(r))
    P = np.zeros((q, r))
    P[piv[:r], :] = Rinv.T
    return K, P, r


def modified_newton_step(grad, hess):
    """
    Compute a safeguarded Newton step using |Lambda| eigenvalues.
    """
    # small ridge for numerical stability
    eps = 1e-8
    hess = hess + eps * np.eye(len(grad))
    Xi, lam, _ = np.linalg.svd(hess)
    H_mod = Xi @ np.diag(np.abs(lam)) @ Xi.T
    return -np.linalg.solve(H_mod + eps * np.eye(len(grad)), grad)


def finite_difference_grad(func, rho, eps=1e-4):
    """Central finite-difference gradient of scalar func at rho."""
    rho = np.asarray(rho, dtype=float)
    grad = np.zeros_like(rho)
    for i in range(len(rho)):
        e = np.zeros_like(rho)
        e[i] = eps
        grad[i] = (func(rho + e) - func(rho - e)) / (2 * eps)
    return grad


def finite_difference_hessian(func, rho, eps=1e-4):
    """Central finite-difference Hessian of scalar func at rho."""
    rho = np.asarray(rho, dtype=float)
    k = len(rho)
    hess = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ei = np.zeros_like(rho)
            ej = np.zeros_like(rho)
            ei[i] = eps
            ej[j] = eps
            fpp = func(rho + ei + ej)
            fpm = func(rho + ei - ej)
            fmp = func(rho - ei + ej)
            fmm = func(rho - ei - ej)
            hess[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps * eps)
    return hess


def compute_dtrA(trA_fn, rho, eps=1e-4):
    """Finite-difference derivative of trace(A) given callable trA_fn(rho)."""
    return finite_difference_grad(trA_fn, rho, eps=eps)


def compute_d2trA(trA_fn, rho, eps=1e-4):
    """Finite-difference Hessian of trace(A)."""
    return finite_difference_hessian(trA_fn, rho, eps=eps)


def compute_dbeta(beta_fn, rho, eps=1e-4):
    """Finite-difference derivative of beta_hat wrt rho. beta_fn -> beta vector."""
    rho = np.asarray(rho, dtype=float)
    base = beta_fn(rho)
    deriv = np.zeros((base.size, rho.size))
    for i in range(rho.size):
        e = np.zeros_like(rho)
        e[i] = eps
        deriv[:, i] = (beta_fn(rho + e) - beta_fn(rho - e)) / (2 * eps)
    return deriv


def compute_d2beta(beta_fn, rho, eps=1e-4):
    """Finite-difference second derivative tensor of beta_hat wrt rho."""
    rho = np.asarray(rho, dtype=float)
    k = rho.size
    base = beta_fn(rho)
    second = np.zeros((base.size, k, k))
    for i in range(k):
        for j in range(k):
            ei = np.zeros_like(rho)
            ej = np.zeros_like(rho)
            ei[i] = eps
            ej[j] = eps
            fpp = beta_fn(rho + ei + ej)
            fpm = beta_fn(rho + ei - ej)
            fmp = beta_fn(rho - ei + ej)
            fmm = beta_fn(rho - ei - ej)
            second[:, i, j] = (fpp - fpm - fmp + fmm) / (4 * eps * eps)
    return second
