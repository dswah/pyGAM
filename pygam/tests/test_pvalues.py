"""Tests for the Wood (2013) smooth-term p-values in pyGAM (issue #163).

Layered from cheap unit checks up to real-data validation:

  liu2 reference distribution
    A  closed-form limits (single/equal/scaled chi-squared)
    B  Wood fractional mixtures vs Monte Carlo
    C  boundary and degenerate inputs
    D  mathematical invariants (monotone, permutation, scale)
    E  estimated-scale quadrature

  _woodteststat statistic
    F  structural invariants / degenerate guards
    G  branch coverage (fractional / integer / rank-deficient)
    H  signal-vs-noise behaviour
    I  scale known vs estimated
    J  Monte-Carlo correctness
    K  curve-space rotation (dropped-R regression guards)
    L  numeric match to compiled mgcv 1.8-41 testStat (12 cases)

  end-to-end (real fits, marked slow)
    M  null calibration (type-I error is Uniform[0,1])
    N  known-signal power and discrimination
    O  tensor / multidimensional terms (incl. mgcv match)
    P  real dataset: smooth + factor terms recover known truth
"""

import numpy as np
import pytest
from scipy import stats

from pygam import LinearGAM, LogisticGAM, f, s, te
from pygam.datasets import mcycle, wage


@pytest.fixture(scope="module")
def gam():
    """A fitted GAM, so we can call its _liu2 / _woodteststat helpers."""
    X, y = mcycle(return_X_y=True)
    return LinearGAM().fit(X, y)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def mc_pvalue(lambdas, x, n_samples=200_000, seed=0):
    """Monte-Carlo P(sum lambda_i chi2_1 > x)."""
    rng = np.random.default_rng(seed)
    lam = np.asarray(lambdas, dtype=float)
    samples = rng.chisquare(1.0, size=(n_samples, len(lam))) @ lam
    return float((samples > x).mean())


def mc_scaled(d, val, k0, n_samples=200_000, seed=0):
    rng = np.random.default_rng(seed)
    val = np.asarray(val, dtype=float)
    Q = rng.chisquare(1.0, size=(n_samples, len(val))) @ val
    Ck = rng.chisquare(k0, size=n_samples)
    return float((Q > d * Ck / k0).mean())


def wood_weights(k_int_part, nu):
    """The reference-distribution weights [1,...,1, nu1, nu2] for a term."""
    rp = nu + 1.0
    nu1 = (rp + np.sqrt(rp * (2.0 - rp))) / 2.0
    return [1.0] * (k_int_part - 1) + [nu1, rp - nu1]


def make_test_Vbj(eigenvalues, seed=0):
    """q x q covariance with the given eigenvalues along random directions."""
    ev = np.asarray(eigenvalues, dtype=float)
    q = len(ev)
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((q, q)))
    return Q @ np.diag(ev) @ Q.T


def make_Xj(q):
    """Model matrix whose column-centred form is orthonormal, so qr(Xj) gives
    R = I and _woodteststat operates directly on the supplied Vbj."""
    n = q + 1
    rng = np.random.default_rng(12345)
    M = np.column_stack([np.ones(n), rng.standard_normal((n, q))])
    Q, _ = np.linalg.qr(M)
    return Q[:, 1 : q + 1]


# ==========================================================================
# A. liu2 -- closed-form limits
# ==========================================================================
@pytest.mark.parametrize("x", [0.5, 2.71, 3.84, 10.83])
def test_liu2_single_chi2_1(gam, x):
    assert abs(gam._liu2(x, [1.0]) - float(stats.chi2.sf(x, df=1))) <= 0.03


@pytest.mark.parametrize("k", [2, 5, 10, 50])
def test_liu2_equal_weights_is_chi2_k(gam, k):
    # sum of k unit-weight chi2_1 == chi2_k; 95th pct must give ~0.05
    x = float(stats.chi2.ppf(0.95, df=k))
    assert abs(gam._liu2(x, [1.0] * k) - 0.05) <= 0.02


@pytest.mark.parametrize("x", [2.0, 8.0])
def test_liu2_scaled_chi2(gam, x):
    assert abs(gam._liu2(x, [2.0, 2.0]) - float(stats.chi2.sf(x / 2.0, df=2))) <= 0.02


# ==========================================================================
# B. liu2 -- Wood fractional mixtures vs Monte Carlo (the real use case)
# ==========================================================================
@pytest.mark.parametrize(
    ("k_int", "nu", "xs"),
    [
        (3, 0.7, [2.0, 5.0, 12.0]),
        (2, 0.3, [1.5, 7.0]),
        (5, 0.9, [3.0, 15.0]),
        (1, 0.5, [0.5, 4.5]),
    ],
)
def test_liu2_wood_mixtures(gam, k_int, nu, xs):
    val = wood_weights(k_int, nu)
    for x in xs:
        exp = mc_pvalue(val, x, seed=int(k_int * 10 + nu * 100 + x))
        assert abs(gam._liu2(x, val) - exp) <= 0.03


@pytest.mark.parametrize(
    "lambdas",
    [
        [10.0] + [0.01] * 20,  # highly skewed
        [np.exp(-i * 0.3) for i in range(15)],  # exponential decay
        [100.0, 0.01],  # bimodal
        [0.95, 0.92, 0.88, 0.5, 0.3, 0.1],
    ],
)
def test_liu2_diverse_configs_vs_mc(gam, lambdas):
    for x in [1.0, 5.0, float(sum(lambdas))]:
        exp = mc_pvalue(lambdas, x, seed=int(x * 7 + 11))
        tol = 0.03 if exp > 0.01 else 0.015
        assert abs(gam._liu2(x, lambdas) - exp) <= tol


def test_liu2_random_fuzz(gam):
    rng = np.random.default_rng(2024)
    for trial in range(8):
        k = int(rng.integers(2, 12))
        lam = rng.uniform(0.05, 3.0, size=k).tolist()
        x = float(rng.uniform(0.5, sum(lam) * 2))
        assert abs(gam._liu2(x, lam) - mc_pvalue(lam, x, seed=trial + 200)) <= 0.03


# ==========================================================================
# C. liu2 -- boundary and degenerate inputs
# ==========================================================================
def test_liu2_boundary_values(gam):
    lam = [1.0, 0.7, 0.4, 0.2]
    assert abs(gam._liu2(0.0, lam) - 1.0) <= 0.05
    assert abs(gam._liu2(-100.0, lam) - 1.0) <= 0.01
    assert abs(gam._liu2(1000.0, lam) - 0.0) <= 0.01


def test_liu2_degenerate(gam):
    assert gam._liu2(1.0, []) == 1.0
    assert abs(gam._liu2(3.84, [1.0]) - 0.05) <= 0.005
    assert gam._liu2(1.0, [0.0] * 3) == 0.0
    assert gam._liu2(-0.5, [0.0] * 3) == 1.0
    assert abs(gam._liu2(1.0, [1e-12] * 100) - 0.0) <= 0.01


# ==========================================================================
# D. liu2 -- mathematical invariants
# ==========================================================================
def test_liu2_monotone_and_in_range(gam):
    lam = [1.0, 0.5, 0.3]
    xs = np.linspace(-5, 15, 40)
    ps = [gam._liu2(x, lam) for x in xs]
    assert all(0.0 <= p <= 1.0 for p in ps)
    assert all(ps[i] >= ps[i + 1] - 1e-9 for i in range(len(ps) - 1))


def test_liu2_scale_equivariance(gam):
    lam = [1.0, 0.5, 0.2]
    c = 4.7
    assert abs(gam._liu2(3.0, lam) - gam._liu2(c * 3.0, [c * l for l in lam])) <= 0.001


def test_liu2_permutation_invariance(gam):
    lam = [1.0, 0.5, 0.2, 0.05]
    a = gam._liu2(2.5, lam)
    assert abs(a - gam._liu2(2.5, lam[::-1])) < 1e-10
    assert abs(a - gam._liu2(2.5, sorted(lam))) < 1e-10


# ==========================================================================
# E. liu2 -- estimated-scale quadrature
# ==========================================================================
@pytest.mark.parametrize("k0", [5, 50, 100])
def test_quadrature_matches_F_single(gam, k0):
    d = 4.0
    assert (
        abs(gam._liu2_scaled_quadrature(d, [1.0], k0) - float(stats.f.sf(d, 1, k0)))
        <= 0.02
    )


@pytest.mark.parametrize(("k", "k0"), [(3, 30), (8, 50)])
def test_quadrature_matches_F_multi(gam, k, k0):
    d = 6.0
    assert (
        abs(
            gam._liu2_scaled_quadrature(d, [1.0] * k, k0)
            - float(stats.f.sf(d / k, k, k0))
        )
        <= 0.025
    )


def test_quadrature_wood_mixture_vs_mc(gam):
    val = wood_weights(3, 0.4)
    assert (
        abs(gam._liu2_scaled_quadrature(5.0, val, 20) - mc_scaled(5.0, val, 20, seed=9))
        <= 0.03
    )


def test_quadrature_converges_to_plain_as_k0_grows(gam):
    val = wood_weights(2, 0.5)
    assert (
        abs(gam._liu2_scaled_quadrature(4.0, val, k0=10_000) - gam._liu2(4.0, val))
        <= 0.005
    )


# ==========================================================================
# F. _woodteststat -- structural invariants / degenerate guards
# ==========================================================================
def test_woodteststat_returns_three_typed_values(gam):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2])
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    T, p, r = gam._woodteststat(coef, Vbj, 3.7, make_Xj(6))
    assert isinstance(T, float) and isinstance(p, float) and isinstance(r, int)
    assert 0.0 <= p <= 1.0


def test_woodteststat_deterministic(gam):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=7)
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    assert gam._woodteststat(coef, Vbj, 3.7, make_Xj(6)) == gam._woodteststat(
        coef, Vbj, 3.7, make_Xj(6)
    )


def test_woodteststat_degenerate_Vbj_gives_pval_one(gam):
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    for Vbj in (np.zeros((6, 6)), make_test_Vbj([1e-20] * 6)):
        T, p, r = gam._woodteststat(coef, Vbj, 3.7, make_Xj(6))
        assert p == 1.0


# ==========================================================================
# G. _woodteststat -- branch coverage
# ==========================================================================
@pytest.mark.parametrize("tau", [0.5, 1.5, 2.3, 3.7, 5.9])
def test_woodteststat_fractional_branch(gam, tau):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2])
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    _, p, _ = gam._woodteststat(coef, Vbj, tau, make_Xj(6))
    assert 0.0 <= p <= 1.0


@pytest.mark.parametrize("tau", [1.0, 3.0, 5.0])
def test_woodteststat_integer_branch(gam, tau):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2])
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    _, p, _ = gam._woodteststat(coef, Vbj, tau, make_Xj(6))
    assert 0.0 <= p <= 1.0


def test_woodteststat_clamps_rank_deficient(gam):
    # only 2 real directions; tau=3.7 asks for 4 -> must clamp the rank
    Vbj = make_test_Vbj([5.0, 3.0, 1e-18, 1e-18, 1e-18, 1e-18])
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    _, p, r = gam._woodteststat(coef, Vbj, 3.7, make_Xj(6))
    assert 0.0 <= p <= 1.0 and r <= 2


# ==========================================================================
# H. _woodteststat -- signal vs noise
# ==========================================================================
def test_woodteststat_big_coef_significant(gam):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=1)
    _, p, _ = gam._woodteststat(
        np.array([10.0, 8.0, 6.0, 4.0, 2.0, 1.0]), Vbj, 3.7, make_Xj(6)
    )
    assert p < 0.05


def test_woodteststat_tiny_coef_not_significant(gam):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=1)
    _, p, _ = gam._woodteststat(np.full(6, 0.01), Vbj, 3.7, make_Xj(6))
    assert p > 0.9


def test_woodteststat_zero_coef_gives_pval_one(gam):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=1)
    T, p, _ = gam._woodteststat(np.zeros(6), Vbj, 3.7, make_Xj(6))
    assert T == 0.0 and p == 1.0


# ==========================================================================
# I. _woodteststat -- scale known vs estimated
# ==========================================================================
def test_woodteststat_estimated_scale_no_smaller_than_known(gam):
    # estimating the scale adds uncertainty -> p must be >= the known-scale p
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=1)
    coef = np.array([3.0, 2.0, -1.5, 1.0, 0.5, -0.3])
    _, p_known, _ = gam._woodteststat(coef, Vbj, 3.7, make_Xj(6))
    _, p_est, _ = gam._woodteststat(coef, Vbj, 3.7, make_Xj(6), res_df=30)
    assert p_est >= p_known - 1e-9


@pytest.mark.parametrize("res_df", [10, 200])
def test_woodteststat_integer_estimated_uses_F(gam, res_df):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=1)
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    _, p, _ = gam._woodteststat(coef, Vbj, 4.0, make_Xj(6), res_df=res_df)
    assert 0.0 <= p <= 1.0


# ==========================================================================
# J. _woodteststat -- Monte-Carlo correctness
# ==========================================================================
def mc_woodteststat_pvalue(gam, Vbj, edf_j, T_obs, Xj, n_sim=20_000, seed=0):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(Vbj + 1e-12 * np.eye(Vbj.shape[0]))
    count = sum(
        gam._woodteststat(L @ rng.standard_normal(Vbj.shape[0]), Vbj, edf_j, Xj)[0]
        > T_obs
        for _ in range(n_sim)
    )
    return count / n_sim


@pytest.mark.parametrize(("tau", "beta_seed"), [(3.7, 4), (2.3, 0)])
def test_woodteststat_matches_monte_carlo(gam, tau, beta_seed):
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2], seed=1)
    Xj = make_Xj(6)
    rng = np.random.default_rng(beta_seed + 1000)
    beta = np.linalg.cholesky(Vbj + 1e-12 * np.eye(6)) @ rng.standard_normal(6)
    T, p, _ = gam._woodteststat(beta, Vbj, tau, Xj)
    mc = mc_woodteststat_pvalue(
        gam, Vbj, tau, T, Xj, n_sim=30_000, seed=beta_seed + 5000
    )
    assert abs(p - mc) < 0.05


# ==========================================================================
# K. _woodteststat -- curve-space rotation (dropped-R regression guards)
# ==========================================================================
def test_woodteststat_rotates_by_R_at_fractional_edf(gam):
    """Regression guard for the dropped-R bug: at fractional edf the statistic
    must depend on the QR rotation of a real basis, so a genuine (non-orthonormal)
    Xj gives a different T than the R = I case. If R were dropped they'd match."""
    rng = np.random.default_rng(0)
    Xj_real = rng.standard_normal((100, 6))
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2])
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    T_real, _, _ = gam._woodteststat(coef, Vbj, 3.7, Xj_real)
    T_ident, _, _ = gam._woodteststat(coef, Vbj, 3.7, make_Xj(6))
    assert not np.isclose(T_real, T_ident, rtol=1e-3)


def test_woodteststat_R_cancels_at_full_rank_edf(gam):
    """Counterpart: at full-rank edf the full inverse is basis-free, so the
    rotation cancels and a real basis matches the R = I case."""
    rng = np.random.default_rng(0)
    Xj_real = rng.standard_normal((100, 6))
    Vbj = make_test_Vbj([5.0, 3.0, 2.0, 1.0, 0.5, 0.2])
    coef = np.array([1.0, -0.5, 0.8, 0.3, -0.2, 0.1])
    T_real, _, _ = gam._woodteststat(coef, Vbj, 6.0, Xj_real)
    T_ident, _, _ = gam._woodteststat(coef, Vbj, 6.0, make_Xj(6))
    assert np.isclose(T_real, T_ident, rtol=1e-6)


# ==========================================================================
# L. _woodteststat -- numeric match to compiled mgcv 1.8-41 testStat
# --------------------------------------------------------------------------
# Reference (stat, pval) produced by mgcv:::testStat on the SAME pre-centred
# inputs. The statistic matches to ~1e-9; the p-value matches within the
# documented Davies (mgcv) vs Liu et al. 2009 (pyGAM) approximation gap, which
# is widest in the small-sample estimated-scale tail. So stat is asserted
# tightly and pval loosely.
# ==========================================================================
def _mgcv_case(specs, master_seed, name):
    rng = np.random.default_rng(master_seed)
    out = {}
    for nm, (n, q), ev, vseed, coef, edf, res_df in specs:
        X = rng.standard_normal((n, q))
        X = X - X.mean(axis=0)
        vrng = np.random.default_rng(vseed)
        Q, _ = np.linalg.qr(vrng.standard_normal((q, q)))
        V = Q @ np.diag(np.asarray(ev, dtype=float)) @ Q.T
        out[nm] = (X, V, np.array(coef), edf, res_df)
    return out[name]


_SPECS_B1 = [
    ("c1_frac_known", (20, 4), [4.0, 2.0, 1.0, 0.5], 1, [1.0, -0.5, 0.8, 0.3], 2.7, -1),
    (
        "c2_frac_estscale",
        (30, 5),
        [5.0, 3.0, 2.0, 1.0, 0.4],
        2,
        [2.0, 1.0, -0.7, 0.5, 0.2],
        3.4,
        25,
    ),
    ("c3_int_full", (25, 4), [3.0, 2.0, 1.0, 0.5], 3, [1.5, -1.0, 0.6, 0.2], 4.0, -1),
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
_SPECS_B2 = [
    (
        "b2_nu_tiny",
        (30, 5),
        [5.0, 3.0, 2.0, 1.0, 0.5],
        10,
        [1.0, -0.5, 0.8, 0.3, -0.2],
        3.05,
        -1,
    ),
    (
        "b2_nu_big",
        (30, 5),
        [5.0, 3.0, 2.0, 1.0, 0.5],
        11,
        [1.0, -0.5, 0.8, 0.3, -0.2],
        3.95,
        -1,
    ),
    ("b2_k_equals_1", (20, 3), [4.0, 1.5, 0.6], 12, [1.2, 0.4, -0.7], 1.4, -1),
    (
        "b2_big_q",
        (150, 9),
        [9.0, 7.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.2],
        13,
        [1.0, -0.8, 0.6, 0.5, -0.4, 0.3, -0.2, 0.1, 0.05],
        6.3,
        -1,
    ),
    (
        "b2_small_resdf",
        (20, 4),
        [4.0, 2.0, 1.0, 0.5],
        14,
        [1.5, -0.6, 0.9, 0.3],
        2.6,
        5,
    ),
    (
        "b2_strong_signal",
        (40, 5),
        [5.0, 3.0, 2.0, 1.0, 0.4],
        15,
        [6.0, 4.0, -3.0, 2.5, 1.5],
        3.4,
        -1,
    ),
]
# (stat, pval) from mgcv 1.8-41 mgcv:::testStat
MGCV_REFERENCE = {
    "c1_frac_known": (1.018550885, 0.7881442045),
    "c2_frac_estscale": (6.112807193, 0.3405266276),
    "c3_int_full": (2.864625904, 0.5807294373),
    "c4_int_trunc": (0.4571566394, 0.9775355139),
    "c5_frac_big": (0.9684885577, 0.9877958744),
    "c6_frac_estscale_sm": (0.109650584, 0.9185588708),
    "b2_nu_tiny": (0.3840214292, 0.9550042987),
    "b2_nu_big": (0.8544565278, 0.9431055405),
    "b2_k_equals_1": (1.136140158, 0.5668342894),
    "b2_big_q": (0.4934476887, 0.9992533887),
    "b2_small_resdf": (1.04390657, 0.7821102297),
    "b2_strong_signal": (38.96929537, 0.003368798371),
}


def _lookup_mgcv_case(name):
    if name.startswith("b2_"):
        return _mgcv_case(_SPECS_B2, 777, name)
    return _mgcv_case(_SPECS_B1, 100, name)


@pytest.mark.parametrize("case", list(MGCV_REFERENCE))
def test_woodteststat_matches_mgcv_stat(gam, case):
    X, V, coef, edf, res_df = _lookup_mgcv_case(case)
    stat_mgcv, _ = MGCV_REFERENCE[case]
    T, _, _ = gam._woodteststat(coef, V, edf, X, res_df)
    assert abs(T - stat_mgcv) < 1e-6


@pytest.mark.parametrize("case", list(MGCV_REFERENCE))
def test_woodteststat_matches_mgcv_pval(gam, case):
    X, V, coef, edf, res_df = _lookup_mgcv_case(case)
    _, pval_mgcv = MGCV_REFERENCE[case]
    _, p, _ = gam._woodteststat(coef, V, edf, X, res_df)
    assert abs(p - pval_mgcv) < 0.05


# ==========================================================================
# M. end-to-end null calibration (type-I error).  SLOW: fits many models.
# --------------------------------------------------------------------------
# Under the null (response independent of the covariate) a correct p-value is
# Uniform[0, 1].  Issue #163's bug produced near-zero p-values on pure noise.
# ==========================================================================
def _null_pvalues(term_factory, family, n_sims, n, seed=0):
    out = []
    for i in range(n_sims):
        rng = np.random.default_rng(seed + i)
        d = 1 if family == "uni" else 2
        X = rng.uniform(0.0, 1.0, size=(n, d))
        if "logit" in family:
            y = rng.integers(0, 2, size=n)  # Bernoulli(0.5), independent of X
            g = LogisticGAM(term_factory())
        else:
            y = rng.standard_normal(n)  # noise, independent of X
            g = LinearGAM(term_factory())
        try:
            g.fit(X if d > 1 else X[:, 0], y)
            p = g.statistics_["p_values"][0]
            if np.isfinite(p):
                out.append(float(p))
        except Exception:
            pass
    return np.asarray(out)


@pytest.mark.slow
def test_null_calibration_gaussian_univariate():
    pv = _null_pvalues(lambda: s(0), "uni", n_sims=150, n=150, seed=0)
    assert 0.40 < pv.mean() < 0.60
    assert (pv < 0.05).mean() < 0.12  # not anti-conservative (the bug)
    assert stats.kstest(pv, "uniform").pvalue > 0.01  # not distinguishable from uniform


@pytest.mark.slow
def test_null_calibration_binary_univariate():
    pv = _null_pvalues(lambda: s(0), "uni_logit", n_sims=150, n=200, seed=100)
    assert 0.35 < pv.mean() < 0.65
    assert (pv < 0.05).mean() < 0.15


@pytest.mark.slow
def test_null_calibration_gaussian_tensor():
    pv = _null_pvalues(lambda: te(0, 1), "tensor", n_sims=120, n=200, seed=7)
    assert 0.35 < pv.mean() < 0.65
    assert (pv < 0.05).mean() < 0.15


# ==========================================================================
# N. end-to-end known-signal power and discrimination.  SLOW.
# ==========================================================================
@pytest.mark.slow
def test_power_gaussian_signal_detected():
    """A real smooth effect must be rejected essentially always."""
    rng = np.random.default_rng(1)
    pv = []
    for _ in range(30):
        x = rng.uniform(0.0, 1.0, 200)
        y = np.sin(2.0 * np.pi * x) + 0.5 * rng.standard_normal(200)
        pv.append(LinearGAM(s(0)).fit(x, y).statistics_["p_values"][0])
    assert (np.asarray(pv) < 0.05).mean() > 0.9


@pytest.mark.slow
def test_discrimination_signal_vs_decoy():
    """z = sin(x) + 0*y + noise : s(x) must fire, decoy s(y) must not."""
    rng = np.random.default_rng(0)
    p_sig, p_dec = [], []
    for _ in range(40):
        x = rng.uniform(0.0, 1.0, 300)
        y = rng.uniform(0.0, 1.0, 300)  # decoy: never enters z
        z = np.sin(2.0 * np.pi * x) + 0.5 * rng.standard_normal(300)
        pv = (
            LinearGAM(s(0) + s(1))
            .fit(np.column_stack([x, y]), z)
            .statistics_["p_values"]
        )
        p_sig.append(pv[0])
        p_dec.append(pv[1])
    p_sig, p_dec = np.asarray(p_sig), np.asarray(p_dec)
    assert (p_sig < 0.05).mean() > 0.9  # signal detected
    assert (p_dec < 0.05).mean() < 0.25  # decoy not over-flagged
    assert p_dec.mean() > 0.25  # decoy roughly uniform


@pytest.mark.slow
def test_amplitude_dose_response():
    """Median p(s(x)) must fall as the true signal amplitude grows."""

    def median_p(amp):
        rng = np.random.default_rng(100)
        ps = []
        for _ in range(20):
            x = rng.uniform(0.0, 1.0, 300)
            z = amp * np.sin(2.0 * np.pi * x) + 0.5 * rng.standard_normal(300)
            ps.append(LinearGAM(s(0)).fit(x, z).statistics_["p_values"][0])
        return float(np.median(ps))

    meds = [median_p(a) for a in (0.0, 0.25, 1.0)]
    assert meds[0] > meds[1] > meds[2]  # monotone decreasing
    assert meds[2] < 1e-3  # strong signal -> tiny p


# ==========================================================================
# O. tensor / multidimensional terms
# ==========================================================================
def _extract_term(g, term_i):
    idx = np.asarray(g.terms.get_coef_indices(term_i))
    idx = idx[idx < len(g.statistics_["edf1_per_coef"])]
    coef = g.coef_[idx]
    Vbj = g.statistics_["cov"][np.ix_(idx, idx)]
    Xj = np.asarray(g._modelmat_train_[:, idx].todense())
    edf = float(g.statistics_["edf1_per_coef"][idx].sum())
    return coef, Vbj, Xj, edf


def test_tensor_statistic_matches_mgcv():
    """A tensor term extracted from a real fit must match compiled mgcv.

    Deterministic fit (seed 7, 5x5 basis); mgcv:::testStat on the same extracted
    (X, V, coef, edf) returned stat = 30.080062, pval = 0.01624483689.
    """
    rng = np.random.default_rng(7)
    n = 150
    a0 = rng.uniform(0, 1, n)
    a1 = rng.uniform(0, 1, n)
    z = np.sin(2 * np.pi * a0) * np.cos(2 * np.pi * a1) + 0.3 * rng.standard_normal(n)
    g = LinearGAM(te(0, 1, n_splines=5)).fit(np.column_stack([a0, a1]), z)
    coef, Vbj, Xj, edf = _extract_term(g, 0)
    T, p, _ = g._woodteststat(coef, Vbj, edf, Xj - Xj.mean(0), -1)
    assert abs(T - 30.080062) < 1e-4
    assert abs(p - 0.01624483689) < 0.02


def test_tensor_curve_space_eigenvalues_consistent():
    """mgcv-independent: eig(R Vbj R^T) must equal the nonzero eig of the curve
    covariance Xc Vbj Xc^T for a real (rank-deficient) tensor basis."""
    rng = np.random.default_rng(0)
    n = 300
    x0 = rng.uniform(0, 1, n)
    x1 = rng.uniform(0, 1, n)
    z = np.sin(2 * np.pi * x0) * np.cos(2 * np.pi * x1) + 0.3 * rng.standard_normal(n)
    g = LinearGAM(te(0, 1)).fit(np.column_stack([x0, x1]), z)
    _, Vbj, Xj, _ = _extract_term(g, 0)
    Xc = Xj - Xj.mean(0)
    _, R = np.linalg.qr(Xc)
    eW = np.sort(np.linalg.eigvalsh((R @ Vbj @ R.T + (R @ Vbj @ R.T).T) / 2))[::-1]
    Vf = (Xc @ Vbj @ Xc.T + (Xc @ Vbj @ Xc.T).T) / 2
    eF = np.sort(np.linalg.eigvalsh(Vf))[::-1][: len(eW)]
    assert np.allclose(eW, eF, atol=1e-8)


@pytest.mark.slow
def test_tensor_power_and_null():
    """te(0,1) must fire on a real 2-D interaction and stay quiet on the null."""
    rng = np.random.default_rng(0)
    p_sig, p_null = [], []
    for i in range(30):
        r = np.random.default_rng(i)
        x0 = r.uniform(0, 1, 300)
        x1 = r.uniform(0, 1, 300)
        z_sig = np.sin(2 * np.pi * x0) * np.cos(
            2 * np.pi * x1
        ) + 0.3 * r.standard_normal(300)
        z_null = r.standard_normal(300)
        X = np.column_stack([x0, x1])
        p_sig.append(LinearGAM(te(0, 1)).fit(X, z_sig).statistics_["p_values"][0])
        p_null.append(LinearGAM(te(0, 1)).fit(X, z_null).statistics_["p_values"][0])
    assert (np.asarray(p_sig) < 0.05).mean() > 0.9
    assert 0.30 < np.asarray(p_null).mean() < 0.70


@pytest.mark.slow
def test_tensor_in_mixed_model():
    """s(0) + te(1,2) where only the tensor carries signal: tensor fires, decoy s(0) doesn't."""
    rng = np.random.default_rng(0)
    fire_te, fire_s = 0, 0
    N = 30
    for i in range(N):
        r = np.random.default_rng(1000 + i)
        x0 = r.uniform(0, 1, 300)
        x1 = r.uniform(0, 1, 300)
        x2 = r.uniform(0, 1, 300)
        y = np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2) + 0.3 * r.standard_normal(
            300
        )
        pv = (
            LinearGAM(s(0) + te(1, 2))
            .fit(np.column_stack([x0, x1, x2]), y)
            .statistics_["p_values"]
        )
        fire_te += pv[1] < 0.05
        fire_s += pv[0] < 0.05
    assert fire_te / N > 0.9
    assert fire_s / N < 0.25


# ==========================================================================
# P. real dataset: smooth + factor terms recover known truth.  SLOW.
# ==========================================================================
@pytest.mark.slow
def test_wage_dataset_recovers_known_effects():
    """ISLR wage data: s(year) weak-but-real, s(age) strong, f(education) strong.
    Exercises the FACTOR-term p-value path end-to-end on real data."""
    X, y = wage(return_X_y=True)
    g = LinearGAM(s(0) + s(1) + f(2)).fit(X, y)
    p_year, p_age, p_edu, p_int = g.statistics_["p_values"]
    assert p_age < 0.05  # age strongly nonlinear
    assert p_edu < 0.05  # education (factor) strong
    assert p_year < 0.05  # year weak but present
    assert np.isnan(p_int)  # intercept: no test
