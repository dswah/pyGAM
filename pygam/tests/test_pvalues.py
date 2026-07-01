"""Tests for Wood (2013) p-value helper functions (liu2)."""

import numpy as np
import pytest
from pygam import LinearGAM
from pygam.datasets import mcycle
from scipy import stats


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
    ("k_int", "nu", "xs"),
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


@pytest.mark.parametrize(("k", "k0"), [(3, 30), (5, 100), (8, 50)])
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
    ("lambdas", "xs"),
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


# =====================================================================
# Tests for Wood (2013) test statistic (_woodteststat)
# =====================================================================
# _woodteststat now takes the term's model matrix Xj and computes the
# statistic in curve space. make_Xj builds an Xj whose column-centered
# form is orthonormal, so qr(Xj) gives R = I and the statistic is
# computed directly on the supplied Vbj (no rotation). This lets each
# unit test specify, via Vbj, exactly the matrix to eigendecompose.


def make_test_Vbj(eigenvalues, seed=0):
    """Build a q x q covariance with the given eigenvalues (random directions)."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    q = len(eigenvalues)
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((q, q))
    Q, _ = np.linalg.qr(A)
    return Q @ np.diag(eigenvalues) @ Q.T


def make_Xj(q):
    """Model matrix whose column-centered form is orthonormal.

    The columns are orthonormal and orthogonal to the constant vector, so
    column-centering is a no-op and qr(Xj) yields R = I. The curve-space
    transform W = R Vbj R^T then reduces to Vbj, so _woodteststat operates
    directly on the supplied Vbj and coefficients.
    """
    n = q + 1
    rng = np.random.default_rng(12345)
    M = np.column_stack([np.ones(n), rng.standard_normal((n, q))])
    Q, _ = np.linalg.qr(M)
    return Q[:, 1 : q + 1]


def mc_woodteststat_pvalue(gam, Vbj, edf_j, T_obs, Xj, n_sim=20_000, seed=0):
    """Monte-Carlo p-value: simulate beta ~ N(0, Vbj), count T_sim > T_obs."""
    rng = np.random.default_rng(seed)
    q = Vbj.shape[0]
    L = np.linalg.cholesky(Vbj + 1e-12 * np.eye(q))
    count = 0
    for _ in range(n_sim):
        beta = L @ rng.standard_normal(q)
        T_sim, _, _ = gam._woodteststat(beta, Vbj, edf_j, Xj)
        if T_sim > T_obs:
            count += 1
    return count / n_sim


# --- Category J: Stage A,B,C structural invariants ---


def test_woodteststat_returns_three_floats(gam):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2])
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    T, p, r = gam._woodteststat(coef, Vbj, 3.7, make_Xj(6))
    assert isinstance(T, float) and isinstance(p, float) and isinstance(r, int)


def test_woodteststat_pval_in_unit_interval(gam):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2])
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    _, p, _ = gam._woodteststat(coef, Vbj, 3.7, make_Xj(6))
    assert 0.0 <= p <= 1.0


def test_woodteststat_degenerate_Vbj_returns_pval_one(gam):
    # all-zero covariance: term is effectively zero, p must be 1
    Vbj = np.zeros((6, 6))
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    T, p, r = gam._woodteststat(coef, Vbj, 3.7, make_Xj(6))
    assert T == 0.0
    assert p == 1.0
    assert r == 1


def test_woodteststat_near_zero_Vbj_returns_pval_one(gam):
    # all eigenvalues 1e-20: still degenerate
    Vbj = make_test_Vbj([1e-20] * 6)
    coef = np.ones(6)
    _, p, _ = gam._woodteststat(coef, Vbj, 3.7, make_Xj(6))
    assert p == 1.0


def test_woodteststat_deterministic(gam):
    # same inputs -> same outputs
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=7)
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    a = gam._woodteststat(coef, Vbj, 3.7, make_Xj(6))
    b = gam._woodteststat(coef, Vbj, 3.7, make_Xj(6))
    assert a == b


# --- Category K: branch coverage ---


@pytest.mark.parametrize("tau", [3.7, 2.3, 5.9, 1.5, 0.5])
def test_woodteststat_fractional_branch(gam, tau):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2])
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    _, p, _ = gam._woodteststat(coef, Vbj, tau, make_Xj(6))
    assert 0.0 <= p <= 1.0


@pytest.mark.parametrize("tau", [1.0, 2.0, 3.0, 4.0, 5.0])
def test_woodteststat_integer_branch(gam, tau):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2])
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    _, p, _ = gam._woodteststat(coef, Vbj, tau, make_Xj(6))
    assert 0.0 <= p <= 1.0


def test_woodteststat_stage_c_clamps_rank_deficient(gam):
    # only 2 real directions exist, tau=3.7 asks for k1=4 -> must clamp
    Vbj = make_test_Vbj([5.0, 3.0, 1e-18, 1e-18, 1e-18, 1e-18])
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    _, p, r = gam._woodteststat(coef, Vbj, 3.7, make_Xj(6))
    assert 0.0 <= p <= 1.0
    assert r <= 2


# --- Category L: signal vs noise behaviour ---


def test_woodteststat_big_coef_significant(gam):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=1)
    coef = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 1.0])
    _, p, _ = gam._woodteststat(coef, Vbj, 3.7, make_Xj(6))
    assert p < 0.05


def test_woodteststat_tiny_coef_not_significant(gam):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=1)
    coef = np.full(6, 0.01)
    _, p, _ = gam._woodteststat(coef, Vbj, 3.7, make_Xj(6))
    assert p > 0.9


def test_woodteststat_zero_coef_pval_one(gam):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=1)
    coef = np.zeros(6)
    T, p, _ = gam._woodteststat(coef, Vbj, 3.7, make_Xj(6))
    assert T == 0.0
    assert p == 1.0


# --- Category M: scale-known vs scale-estimated ---


def test_woodteststat_estimated_scale_runs(gam):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=1)
    coef = np.array([2.0, 1.5, -1.0, 0.8, 0.3, -0.2])
    _, p, _ = gam._woodteststat(coef, Vbj, 3.7, make_Xj(6), res_df=80)
    assert 0.0 <= p <= 1.0


def test_woodteststat_estimated_scale_no_smaller_than_known(gam):
    # estimating the scale adds uncertainty -> p should be >= the known-scale one
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=1)
    coef = np.array([3.0, 2.0, -1.5, 1.0, 0.5, -0.3])
    _, p_known, _ = gam._woodteststat(coef, Vbj, 3.7, make_Xj(6))
    _, p_est, _ = gam._woodteststat(coef, Vbj, 3.7, make_Xj(6), res_df=30)
    assert p_est >= p_known - 1e-9


@pytest.mark.parametrize("res_df", [10, 50, 200])
def test_woodteststat_integer_estimated_uses_F(gam, res_df):
    # integer tau, estimated scale -> F distribution path
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=1)
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    _, p, _ = gam._woodteststat(coef, Vbj, 4.0, make_Xj(6), res_df=res_df)
    assert 0.0 <= p <= 1.0


# --- Category N: Monte Carlo validation (statistical correctness) ---


@pytest.mark.parametrize(("tau", "beta_seed"), [(3.7, 4), (2.3, 0), (5.5, 2)])
def test_woodteststat_matches_monte_carlo_fractional(gam, tau, beta_seed):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=1)
    Xj = make_Xj(6)
    rng = np.random.default_rng(beta_seed + 1000)
    beta = np.linalg.cholesky(Vbj + 1e-12 * np.eye(6)) @ rng.standard_normal(6)
    T, p, _ = gam._woodteststat(beta, Vbj, tau, Xj)
    mc = mc_woodteststat_pvalue(
        gam, Vbj, tau, T, Xj, n_sim=30_000, seed=beta_seed + 5000
    )
    assert abs(p - mc) < 0.05


def test_woodteststat_matches_monte_carlo_integer(gam):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=3)
    Xj = make_Xj(6)
    beta = np.array([1.5, 1.0, -0.8, 0.5, 0.2, -0.1])
    T, p, _ = gam._woodteststat(beta, Vbj, 4.0, Xj)
    mc = mc_woodteststat_pvalue(gam, Vbj, 4.0, T, Xj, n_sim=30_000, seed=4)
    assert abs(p - mc) < 0.05


# --- Category O: real (non-orthonormal) basis exercises the curve-space path ---


def test_woodteststat_uses_R_rotation_real_basis(gam):
    """Regression test for the dropped-R bug.

    The statistic is defined in curve space: it must rotate Vbj and coef by R
    from qr(Xj), i.e. eigendecompose W = R Vbj R^T, not Vbj directly. With a
    real (non-orthonormal) basis the rotation changes the answer, so a real Xj
    must give a different statistic than the no-rotation (R = I) case obtained
    from make_Xj. If R were dropped (Vbj eigendecomposed directly, ignoring Xj)
    the two would be identical and this test would fail.

    This only manifests at FRACTIONAL edf: at integer edf the full inverse is
    basis-free, R cancels, and the curve-space and raw-Vbj statistics coincide.
    """
    rng = np.random.default_rng(0)
    Xj_real = rng.standard_normal((100, 6))  # genuine basis -> R != I
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2])
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])

    T_real, _, _ = gam._woodteststat(coef, Vbj, 3.7, Xj_real)  # rotated (curve space)
    T_identity, _, _ = gam._woodteststat(coef, Vbj, 3.7, make_Xj(6))  # R = I -> raw Vbj

    # the rotation must actually change the statistic
    assert not np.isclose(T_real, T_identity, rtol=1e-3)


def test_woodteststat_full_rank_edf_R_cancels(gam):
    """Counterpart: only at FULL-RANK edf (tau == q, every direction inverted)
    does the rotation cancel, so a real basis and the no-rotation case give the
    same statistic (the full inverse is basis-free). At fractional or truncated
    edf the rotation does not cancel - see the test above.
    """
    rng = np.random.default_rng(0)
    Xj_real = rng.standard_normal((100, 6))
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2])
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])

    # tau == q == 6: full inverse, basis-free
    T_real, _, _ = gam._woodteststat(coef, Vbj, 6.0, Xj_real)
    T_identity, _, _ = gam._woodteststat(coef, Vbj, 6.0, make_Xj(6))

    assert np.isclose(T_real, T_identity, rtol=1e-6)


def test_woodteststat_real_basis_valid_output(gam):
    """A genuine non-orthonormal basis still returns a well-formed result."""
    rng = np.random.default_rng(1)
    Xj = rng.standard_normal((100, 6))
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2])
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    T, p, r = gam._woodteststat(coef, Vbj, 3.7, Xj)
    assert T >= 0.0
    assert 0.0 <= p <= 1.0
    assert isinstance(r, int)


# --- Category P: validation against compiled mgcv (mgcv 1.8-41 testStat) ---
#
# The reference stat/pval below were produced by running mgcv:::testStat on the
# exact same inputs (real non-orthonormal bases, fractional and integer edf,
# known and estimated scale). The bases are pre-centered so pyGAM's internal
# column-centering is a no-op and both QR the identical matrix.
#
# The statistic matches mgcv to ~1e-9 across every case. The p-value matches
# closely except a wider gap on the small-sample estimated-scale case (c6):
# mgcv's psum.chisq uses Davies' method while pyGAM uses the Liu et al. (2009)
# approximation, and the two disagree most in that tail. Hence stat is checked
# tightly and pval loosely.


def _mgcv_case(name):
    """Regenerate the exact (X, V, coef, edf, res_df) validated against mgcv.

    PCG64 (np.random.default_rng) is version-stable, so these reproduce the
    arrays mgcv:::testStat was evaluated on bit-for-bit.
    """
    specs = [
        (
            "c1_frac_known",
            (20, 4),
            [4.0, 2.0, 1.0, 0.5],
            1,
            [1.0, -0.5, 0.8, 0.3],
            2.7,
            -1,
        ),
        (
            "c2_frac_estscale",
            (30, 5),
            [5.0, 3.0, 2.0, 1.0, 0.4],
            2,
            [2.0, 1.0, -0.7, 0.5, 0.2],
            3.4,
            25,
        ),
        (
            "c3_int_full",
            (25, 4),
            [3.0, 2.0, 1.0, 0.5],
            3,
            [1.5, -1.0, 0.6, 0.2],
            4.0,
            -1,
        ),
        (
            "c4_int_trunc",
            (40, 6),
            [6.0, 4.0, 3.0, 2.0, 1.0, 0.5],
            4,
            [1.0, -0.5, 0.8, 0.3, -0.2, 0.1],
            4.0,
            -1,
        ),
        (
            "c5_frac_big",
            (100, 8),
            [8.0, 6.0, 5.0, 3.0, 2.0, 1.0, 0.6, 0.3],
            5,
            [1.2, -0.8, 1.0, 0.5, -0.3, 0.4, 0.1, -0.05],
            5.6,
            -1,
        ),
        ("c6_frac_estscale_sm", (15, 3), [3.0, 1.5, 0.7], 6, [1.0, 0.5, -0.3], 1.8, 12),
    ]
    rng = np.random.default_rng(100)
    out = {}
    for nm, (n, q), ev, vseed, coef, edf, res_df in specs:
        X = rng.standard_normal((n, q))
        X = X - X.mean(axis=0)
        vrng = np.random.default_rng(vseed)
        Q, _ = np.linalg.qr(vrng.standard_normal((q, q)))
        V = Q @ np.diag(np.asarray(ev, dtype=float)) @ Q.T
        out[nm] = (X, V, np.array(coef), edf, res_df)
    return out[name]


# (stat, pval) from mgcv 1.8-41 mgcv:::testStat
MGCV_REFERENCE = {
    "c1_frac_known": (1.018550885, 0.7881442045),
    "c2_frac_estscale": (6.112807193, 0.3405266276),
    "c3_int_full": (2.864625904, 0.5807294373),
    "c4_int_trunc": (0.4571566394, 0.9775355139),
    "c5_frac_big": (0.9684885577, 0.9877958744),
    "c6_frac_estscale_sm": (0.109650584, 0.9185588708),
}


@pytest.mark.parametrize("case", list(MGCV_REFERENCE))
def test_woodteststat_matches_mgcv_stat(gam, case):
    """The test statistic must match compiled mgcv to high precision."""
    X, V, coef, edf, res_df = _mgcv_case(case)
    stat_mgcv, _ = MGCV_REFERENCE[case]
    T, _, _ = gam._woodteststat(coef, V, edf, X, res_df)
    assert abs(T - stat_mgcv) < 1e-6


@pytest.mark.parametrize("case", list(MGCV_REFERENCE))
def test_woodteststat_matches_mgcv_pval(gam, case):
    """The p-value must match mgcv within the Davies-vs-Liu approximation gap."""
    X, V, coef, edf, res_df = _mgcv_case(case)
    _, pval_mgcv = MGCV_REFERENCE[case]
    _, p, _ = gam._woodteststat(coef, V, edf, X, res_df)
    assert abs(p - pval_mgcv) < 0.05
