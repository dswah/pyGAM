"""Tests for Wood (2013) p-value helper functions (liu2)."""

import numpy as np
import pytest
from scipy import stats

from pygam import LinearGAM
from pygam.datasets import mcycle


@pytest.fixture(scope="module")
def gam():
    """A fitted GAM instance, so we can call its _liu2 methods."""
    X, y = mcycle(return_X_y=True)
    return LinearGAM().fit(X, y)


def mc_pvalue(lambdas, x, n_samples=300_000, seed=0):
    rng = np.random.default_rng(seed)
    lam = np.asarray(lambdas, dtype=float)
    samples = rng.chisquare(1.0, size=(n_samples, len(lam))) @ lam
    return float((samples > x).mean())


def mc_scaled(d, val, k0, n_samples=300_000, seed=0):
    rng = np.random.default_rng(seed)
    val = np.asarray(val, dtype=float)
    Q = rng.chisquare(1.0, size=(n_samples, len(val))) @ val
    Ck = rng.chisquare(k0, size=n_samples)
    return float((Q > d * Ck / k0).mean())


def wood_weights(k_int_part, nu):
    rp = nu + 1.0
    nu1 = (rp + np.sqrt(rp * (2.0 - rp))) / 2.0
    nu2 = rp - nu1
    return [1.0] * (k_int_part - 1) + [nu1, nu2]


# --- Category A: closed-form standard cases ---


@pytest.mark.parametrize("x", [0.5, 1.0, 2.71, 3.84, 6.63, 10.83])
def test_liu2_single_chi2_1(gam, x):
    got = gam._liu2(x, [1.0])
    expected = float(stats.chi2.sf(x, df=1))
    assert abs(got - expected) <= 0.03


@pytest.mark.parametrize("k", [2, 3, 5, 10])
def test_liu2_equal_weights_chi2_k(gam, k):
    x = float(stats.chi2.ppf(0.95, df=k))
    got = gam._liu2(x, [1.0] * k)
    assert abs(got - 0.05) <= 0.02


@pytest.mark.parametrize("x", [2.0, 4.0, 8.0])
def test_liu2_scaled_chi2(gam, x):
    got = gam._liu2(x, [2.0, 2.0])
    expected = float(stats.chi2.sf(x / 2.0, df=2))
    assert abs(got - expected) <= 0.02


# --- Category B: Wood-style fractional mixtures (vs Monte Carlo) ---


@pytest.mark.parametrize(
    "k_int,nu,xs",
    [
        (3, 0.7, [2.0, 5.0, 7.5, 12.0]),
        (2, 0.3, [1.5, 4.0, 7.0]),
        (5, 0.9, [3.0, 8.0, 15.0]),
        (1, 0.5, [0.5, 2.0, 4.5]),
    ],
)
def test_liu2_wood_mixtures(gam, k_int, nu, xs):
    val = wood_weights(k_int, nu)
    for x in xs:
        got = gam._liu2(x, val)
        expected = mc_pvalue(val, x, seed=int(k_int * 10 + nu * 100 + x))
        assert abs(got - expected) <= 0.025


# --- Category C: boundary x values ---


def test_liu2_boundary(gam):
    lambdas = [1.0, 0.7, 0.4, 0.2]
    assert abs(gam._liu2(0.0, lambdas) - 1.0) <= 0.05
    assert abs(gam._liu2(-100.0, lambdas) - 1.0) <= 0.01
    assert abs(gam._liu2(1000.0, lambdas) - 0.0) <= 0.01
    muQ = sum(lambdas)
    assert abs(gam._liu2(muQ, lambdas) - mc_pvalue(lambdas, muQ, seed=5)) <= 0.03


# --- Category D: degenerate inputs ---


def test_liu2_degenerate(gam):
    assert gam._liu2(1.0, []) == 1.0
    assert abs(gam._liu2(3.84, [1.0]) - 0.05) <= 0.005
    assert gam._liu2(1.0, [0.0] * 3) == 0.0
    assert gam._liu2(-0.5, [0.0] * 3) == 1.0
    assert abs(gam._liu2(1.0, [1e-12] * 100) - 0.0) <= 0.01


# --- Category E: stress / extreme parameters ---


@pytest.mark.parametrize("x", [1.0, 10.0, 30.0])
def test_liu2_skewed(gam, x):
    lambdas = [10.0] + [0.01] * 20
    assert abs(gam._liu2(x, lambdas) - mc_pvalue(lambdas, x, seed=6)) <= 0.03


def test_liu2_high_dim(gam):
    k = 50
    x = float(stats.chi2.ppf(0.95, df=k))
    assert abs(gam._liu2(x, [1.0] * k) - 0.05) <= 0.02


@pytest.mark.parametrize("x", [0.5, 2.0, 5.0])
def test_liu2_exp_decay(gam, x):
    lambdas = [np.exp(-i * 0.3) for i in range(15)]
    assert abs(gam._liu2(x, lambdas) - mc_pvalue(lambdas, x, seed=7)) <= 0.03


@pytest.mark.parametrize("x", [50.0, 150.0, 500.0])
def test_liu2_bimodal(gam, x):
    lambdas = [100.0, 0.01]
    assert abs(gam._liu2(x, lambdas) - mc_pvalue(lambdas, x, seed=8)) <= 0.03


# --- Category F: internal consistency ---


def test_liu2_monotonic(gam):
    lambdas = [1.0, 0.5, 0.3]
    xs = np.linspace(0.5, 10, 20)
    pvals = [gam._liu2(x, lambdas) for x in xs]
    assert all(pvals[i] >= pvals[i + 1] - 1e-9 for i in range(len(pvals) - 1))


def test_liu2_in_range(gam):
    xs = np.concatenate([np.linspace(-10, 0, 5), np.linspace(0.1, 50, 20)])
    assert all(0 <= gam._liu2(x, [1.0, 0.7, 0.4, 0.2]) <= 1 for x in xs)


def test_liu2_scale_equivariance(gam):
    lambdas = [1.0, 0.5, 0.2]
    c = 4.7
    got1 = gam._liu2(3.0, lambdas)
    got2 = gam._liu2(c * 3.0, [c * l for l in lambdas])
    assert abs(got1 - got2) <= 0.001


def test_liu2_permutation_invariance(gam):
    lambdas = [1.0, 0.5, 0.2, 0.05]
    a = gam._liu2(2.5, lambdas)
    b = gam._liu2(2.5, list(reversed(lambdas)))
    c = gam._liu2(2.5, sorted(lambdas))
    assert abs(a - b) < 1e-10 and abs(a - c) < 1e-10


# --- Category G: scaled quadrature (estimated-scale case) ---


@pytest.mark.parametrize("k0", [5, 20, 50, 100])
def test_quadrature_f_1_k0(gam, k0):
    d = 4.0
    got = gam._liu2_scaled_quadrature(d, [1.0], k0)
    expected = float(stats.f.sf(d, dfn=1, dfd=k0))
    assert abs(got - expected) <= 0.02


@pytest.mark.parametrize("k,k0", [(3, 30), (5, 100), (8, 50)])
def test_quadrature_f_k_k0(gam, k, k0):
    d = 6.0
    got = gam._liu2_scaled_quadrature(d, [1.0] * k, k0)
    expected = float(stats.f.sf(d / k, dfn=k, dfd=k0))
    assert abs(got - expected) <= 0.025


@pytest.mark.parametrize("k0", [20, 100])
def test_quadrature_wood_mixture(gam, k0):
    val = wood_weights(3, 0.4)
    d = 5.0
    got = gam._liu2_scaled_quadrature(d, val, k0)
    expected = mc_scaled(d, val, k0, seed=9)
    assert abs(got - expected) <= 0.03


def test_quadrature_converges_to_plain(gam):
    val = wood_weights(2, 0.5)
    plain = gam._liu2(4.0, val)
    scaled = gam._liu2_scaled_quadrature(4.0, val, k0=10_000)
    assert abs(scaled - plain) <= 0.005


# --- Category H: heavy Monte Carlo ---


@pytest.mark.parametrize(
    "lambdas,xs",
    [
        ([1.0, 0.5, 0.3, 0.1], [0.5, 1.5, 3.0, 6.0, 10.0]),
        ([2.0, 1.5, 1.0, 0.5, 0.2], [1.0, 4.0, 8.0, 14.0]),
        (wood_weights(3, 0.7), [2.0, 5.0, 8.0, 15.0]),
        (wood_weights(5, 0.2), [3.0, 7.0, 12.0, 20.0]),
        ([0.95, 0.92, 0.88, 0.85, 0.5, 0.3, 0.1], [1.0, 3.0, 6.0, 10.0]),
    ],
)
def test_liu2_monte_carlo(gam, lambdas, xs):
    for x in xs:
        got = gam._liu2(x, lambdas)
        expected = mc_pvalue(lambdas, x, n_samples=300_000, seed=int(x * 7 + 11))
        tol = 0.025 if expected > 0.01 else 0.01
        assert abs(got - expected) <= tol


# --- Category I: random configurations ---


def test_liu2_random_configs(gam):
    rng = np.random.default_rng(2024)
    for trial in range(15):
        k = int(rng.integers(2, 12))
        lambdas = rng.uniform(0.05, 3.0, size=k).tolist()
        x = float(rng.uniform(0.5, sum(lambdas) * 2))
        got = gam._liu2(x, lambdas)
        expected = mc_pvalue(lambdas, x, seed=trial + 200)
        assert abs(got - expected) <= 0.03


"""Tests for Wood (2013) p-value helper function (_woodteststat)."""


def make_test_Vbj(eigenvalues, seed=0):
    """Build a q x q covariance with the given eigenvalues (random directions)."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    q = len(eigenvalues)
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((q, q))
    Q, _ = np.linalg.qr(A)
    return Q @ np.diag(eigenvalues) @ Q.T


def mc_woodteststat_pvalue(gam, Vbj, edf_j, T_obs, n_sim=20_000, seed=0):
    """Monte-Carlo p-value: simulate beta ~ N(0, Vbj), count T_sim > T_obs."""
    rng = np.random.default_rng(seed)
    q = Vbj.shape[0]
    L = np.linalg.cholesky(Vbj + 1e-12 * np.eye(q))
    count = 0
    for _ in range(n_sim):
        beta = L @ rng.standard_normal(q)
        T_sim, _, _ = gam._woodteststat(beta, Vbj, edf_j)
        if T_sim > T_obs:
            count += 1
    return count / n_sim


# --- Category J: Stage A,B,C structural invariants ---


def test_woodteststat_returns_three_floats(gam):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2])
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    T, p, r = gam._woodteststat(coef, Vbj, 3.7)
    assert isinstance(T, float) and isinstance(p, float) and isinstance(r, int)


def test_woodteststat_pval_in_unit_interval(gam):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2])
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    _, p, _ = gam._woodteststat(coef, Vbj, 3.7)
    assert 0.0 <= p <= 1.0


def test_woodteststat_degenerate_Vbj_returns_pval_one(gam):
    # all-zero covariance: term is effectively zero, p must be 1
    Vbj = np.zeros((6, 6))
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    T, p, r = gam._woodteststat(coef, Vbj, 3.7)
    assert T == 0.0
    assert p == 1.0
    assert r == 1


def test_woodteststat_near_zero_Vbj_returns_pval_one(gam):
    # all eigenvalues 1e-20: still degenerate
    Vbj = make_test_Vbj([1e-20] * 6)
    coef = np.ones(6)
    _, p, _ = gam._woodteststat(coef, Vbj, 3.7)
    assert p == 1.0


def test_woodteststat_deterministic(gam):
    # same inputs -> same outputs
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=7)
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    a = gam._woodteststat(coef, Vbj, 3.7)
    b = gam._woodteststat(coef, Vbj, 3.7)
    assert a == b


# --- Category K: branch coverage ---


@pytest.mark.parametrize("tau", [3.7, 2.3, 5.9, 1.5, 0.5])
def test_woodteststat_fractional_branch(gam, tau):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2])
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    _, p, _ = gam._woodteststat(coef, Vbj, tau)
    assert 0.0 <= p <= 1.0


@pytest.mark.parametrize("tau", [1.0, 2.0, 3.0, 4.0, 5.0])
def test_woodteststat_integer_branch(gam, tau):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2])
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    _, p, _ = gam._woodteststat(coef, Vbj, tau)
    assert 0.0 <= p <= 1.0


def test_woodteststat_stage_c_clamps_rank_deficient(gam):
    # only 2 real directions exist, tau=3.7 asks for k1=4 -> must clamp
    Vbj = make_test_Vbj([5.0, 3.0, 1e-18, 1e-18, 1e-18, 1e-18])
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    _, p, r = gam._woodteststat(coef, Vbj, 3.7)
    assert 0.0 <= p <= 1.0
    assert r <= 2


# --- Category L: signal vs noise behaviour ---


def test_woodteststat_big_coef_significant(gam):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=1)
    coef = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 1.0])
    _, p, _ = gam._woodteststat(coef, Vbj, 3.7)
    assert p < 0.05


def test_woodteststat_tiny_coef_not_significant(gam):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=1)
    coef = np.full(6, 0.01)
    _, p, _ = gam._woodteststat(coef, Vbj, 3.7)
    assert p > 0.9


def test_woodteststat_zero_coef_pval_one(gam):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=1)
    coef = np.zeros(6)
    T, p, _ = gam._woodteststat(coef, Vbj, 3.7)
    assert T == 0.0
    assert p == 1.0


# --- Category M: scale-known vs scale-estimated ---


def test_woodteststat_estimated_scale_runs(gam):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=1)
    coef = np.array([2.0, 1.5, -1.0, 0.8, 0.3, -0.2])
    _, p, _ = gam._woodteststat(coef, Vbj, 3.7, res_df=80)
    assert 0.0 <= p <= 1.0


def test_woodteststat_estimated_scale_no_smaller_than_known(gam):
    # estimating the scale adds uncertainty -> p should be >= the known-scale one
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=1)
    coef = np.array([3.0, 2.0, -1.5, 1.0, 0.5, -0.3])
    _, p_known, _ = gam._woodteststat(coef, Vbj, 3.7)
    _, p_est, _ = gam._woodteststat(coef, Vbj, 3.7, res_df=30)
    assert p_est >= p_known - 1e-9


@pytest.mark.parametrize("res_df", [10, 50, 200])
def test_woodteststat_integer_estimated_uses_F(gam, res_df):
    # integer tau, estimated scale -> F distribution path
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=1)
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    _, p, _ = gam._woodteststat(coef, Vbj, 4.0, res_df=res_df)
    assert 0.0 <= p <= 1.0


# --- Category N: Monte Carlo validation (statistical correctness) ---


@pytest.mark.parametrize("tau,beta_seed", [(3.7, 4), (2.3, 0), (5.5, 2)])
def test_woodteststat_matches_monte_carlo_fractional(gam, tau, beta_seed):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=1)
    rng = np.random.default_rng(beta_seed + 1000)
    beta = np.linalg.cholesky(Vbj + 1e-12 * np.eye(6)) @ rng.standard_normal(6)
    T, p, _ = gam._woodteststat(beta, Vbj, tau)
    mc = mc_woodteststat_pvalue(gam, Vbj, tau, T, n_sim=30_000, seed=beta_seed + 5000)
    assert abs(p - mc) < 0.05


def test_woodteststat_matches_monte_carlo_integer(gam):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=3)
    beta = np.array([1.5, 1.0, -0.8, 0.5, 0.2, -0.1])
    T, p, _ = gam._woodteststat(beta, Vbj, 4.0)
    mc = mc_woodteststat_pvalue(gam, Vbj, 4.0, T, n_sim=30_000, seed=4)
    assert abs(p - mc) < 0.05
