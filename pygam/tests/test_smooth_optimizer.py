import numpy as np

from pygam import LinearGAM, LogisticGAM, PoissonGAM, s
from pygam.distributions import BinomialDist, GammaDist, PoissonDist
from pygam.links import LogLink, LogitLink
from pygam.smooth_optimizer import (
    compute_dtrA,
    compute_d2trA,
    finite_difference_grad,
    finite_difference_hessian,
    modified_newton_step,
    qr_decomp,
)


def test_distribution_v_prime():
    for Dist in [PoissonDist, BinomialDist, GammaDist]:
        dist = Dist()
        mu = np.linspace(0.1, 0.9, 10)
        analytical = dist.V_prime(mu)
        numerical = (dist.V(mu + 1e-5) - dist.V(mu - 1e-5)) / (2e-5)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)


def test_link_g_double_prime():
    for Link in [LogLink, LogitLink]:
        link = Link()
        mu = np.linspace(0.1, 0.9, 10)
        analytical = link.g_double_prime(mu, getattr(link, "dist", None) or BinomialDist())
        numerical = (
            link.link(mu + 1e-5, getattr(link, "dist", None) or BinomialDist())
            - 2 * link.link(mu, getattr(link, "dist", None) or BinomialDist())
            + link.link(mu - 1e-5, getattr(link, "dist", None) or BinomialDist())
        ) / (1e-10)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-3, atol=1e-5)


def test_qr_consistency():
    rng = np.random.default_rng(42)
    n, q = 60, 6
    X = rng.standard_normal((n, q))
    W = np.abs(rng.standard_normal(n)) + 0.1
    S = np.eye(q) * 0.5
    eigvals, eigvecs = np.linalg.eigh(S)
    E = (eigvecs * np.sqrt(np.maximum(eigvals, 0))).T
    K, P, r = qr_decomp(W[:, None] * X, E)
    G = X.T @ np.diag(W**2) @ X + S
    G_inv = np.linalg.inv(G)
    PPt = P @ P.T
    np.testing.assert_allclose(G_inv, PPt, rtol=1e-1, atol=1e-2)
    assert r == q


def test_finite_difference_helpers():
    f = lambda r: np.sum(r**2)
    rho = np.array([0.1, -0.2])
    g = finite_difference_grad(f, rho, eps=1e-5)
    h = finite_difference_hessian(f, rho, eps=1e-5)
    np.testing.assert_allclose(g, 2 * rho, rtol=1e-4)
    np.testing.assert_allclose(h, np.eye(2) * 2, rtol=1e-3)
    step = modified_newton_step(g, h)
    np.testing.assert_allclose(step, -rho, rtol=1e-3)


def test_auto_smooth_gaussian():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (120, 1))
    y = np.sin(2 * np.pi * X[:, 0]) + rng.normal(scale=0.3, size=120)
    gam_grid = LinearGAM(s(0)).gridsearch(X, y, lam=np.logspace(-2, 2, 5))
    gam_auto = LinearGAM(s(0)).fit_auto_smooth(X, y, criterion="gcv", max_iter_outer=6)
    gcv_grid = gam_grid.statistics_["GCV"]
    gcv_auto = gam_auto.statistics_["GCV"]
    assert gcv_auto <= gcv_grid * 1.1


def test_auto_smooth_poisson():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (150, 1))
    y = rng.poisson(np.exp(np.sin(2 * np.pi * X[:, 0])) * 2.0)
    gam = PoissonGAM(s(0)).fit_auto_smooth(X, y, criterion="aic", max_iter_outer=6)
    assert gam.statistics_["AIC"] is not None
    assert np.all(np.asarray(gam.statistics_["lam"]) > 0)


def test_auto_smooth_logistic_concurvity():
    rng = np.random.default_rng(2)
    n = 200
    x = rng.uniform(size=n)
    d = x**3 + rng.normal(scale=0.01, size=n)
    f_true = (d - 0.5 + 10 * (d - 0.5) ** 3) * 5
    prob = 1 / (1 + np.exp(-f_true))
    y = rng.binomial(1, prob)
    X = np.column_stack([x, d])
    gam = LogisticGAM(s(0) + s(1)).fit_auto_smooth(X, y, criterion="aic", max_iter_outer=5)
    assert gam._is_fitted
