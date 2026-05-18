import warnings

import numpy as np

from pygam import LinearGAM, f, s

# --- DATA GENERATORS ---


def gen_noise_univariate(rng):
    """Standard Univariate Noise"""
    n = 200
    X = rng.standard_normal((n, 1))
    y = rng.standard_normal(n)
    return X, y


def gen_strong_signal(rng):
    """Strong Sine Wave"""
    n = 200
    X = rng.standard_normal((n, 1))
    y = np.sin(X * 2).flatten() + rng.standard_normal(n) * 0.5
    return X, y


def gen_multivariate_noise(rng):
    """Two uncorrelated noise predictors"""
    n = 200
    X = rng.standard_normal((n, 2))
    y = rng.standard_normal(n)
    return X, y


def gen_mixed_data(rng):
    """Continuous + Categorical (3 levels)"""
    n = 200
    x_cont = rng.standard_normal((n, 1))
    x_cat = rng.choice([0, 1, 2], size=(n, 1))
    X = np.hstack([x_cont, x_cat])
    y = rng.standard_normal(n)  # Pure noise response
    return X, y


def gen_collinear_data(rng):
    """Two highly correlated predictors"""
    n = 200
    x1 = rng.standard_normal((n, 1))
    x2 = x1 + rng.standard_normal((n, 1)) * 0.001  # Almost identical
    X = np.hstack([x1, x2])
    y = rng.standard_normal(n)
    return X, y


# --- TEST ENGINE ---

N_SIMS = 100
SEED = 12345
FPR_BOUNDS = (1.0, 10.0)  # window around the 5% nominal level


def run_test_scenario(data_gen_func, gam_factory, term_idx=0, n_sims=N_SIMS, seed=SEED):
    """Run a simulation loop and return the rejection rate (%) for the term.

    The RNG is seeded so that the loop is reproducible across runs.
    Fits that fail to converge are skipped; if every fit fails the test
    will raise rather than silently report 0%.
    """
    rng = np.random.default_rng(seed)
    rejections = 0
    n_fitted = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_sims):
            X, y = data_gen_func(rng)
            try:
                gam = gam_factory().fit(X, y)
            except Exception:  # noqa: S112, BLE001
                continue
            n_fitted += 1
            if gam.statistics_["p_values"][term_idx] < 0.05:
                rejections += 1

    if n_fitted == 0:
        raise RuntimeError("no GAM fits succeeded in this scenario")

    return (rejections / n_fitted) * 100


# --- PYTEST CASES ---


def test_fpr_noise_unpenalized():
    """Univariate noise, lam=0. FPR should be near 5%."""
    factory = lambda: LinearGAM(s(0, n_splines=10), lam=0)
    rate = run_test_scenario(gen_noise_univariate, factory)
    assert FPR_BOUNDS[0] <= rate <= FPR_BOUNDS[1], (
        f"FPR {rate:.1f}% outside {FPR_BOUNDS}"
    )


def test_fpr_noise_smoothed():
    """Univariate noise, lam=0.6. FPR should be near 5%."""
    factory = lambda: LinearGAM(s(0, n_splines=10), lam=0.6)
    rate = run_test_scenario(gen_noise_univariate, factory)
    assert FPR_BOUNDS[0] <= rate <= FPR_BOUNDS[1], (
        f"FPR {rate:.1f}% outside {FPR_BOUNDS}"
    )


def test_power_strong_signal():
    """Strong signal. Power should be near 100%."""
    factory = lambda: LinearGAM(s(0, n_splines=10), lam=0.6)
    rate = run_test_scenario(gen_strong_signal, factory)
    assert rate >= 95.0, f"Power {rate:.1f}% is too low (Target >=95%)"


def test_multivariate_term0():
    """Multivariate s(0) + s(1), check term 0."""
    factory = lambda: LinearGAM(s(0) + s(1), lam=0.6)
    rate = run_test_scenario(gen_multivariate_noise, factory, term_idx=0)
    assert FPR_BOUNDS[0] <= rate <= FPR_BOUNDS[1], (
        f"FPR {rate:.1f}% outside {FPR_BOUNDS}"
    )


def test_multivariate_term1():
    """Multivariate s(0) + s(1), check term 1."""
    factory = lambda: LinearGAM(s(0) + s(1), lam=0.6)
    rate = run_test_scenario(gen_multivariate_noise, factory, term_idx=1)
    assert FPR_BOUNDS[0] <= rate <= FPR_BOUNDS[1], (
        f"FPR {rate:.1f}% outside {FPR_BOUNDS}"
    )


def test_mixed_types():
    """Spline + factor, check the spline term."""
    factory = lambda: LinearGAM(s(0) + f(1), lam=0.6)
    rate = run_test_scenario(gen_mixed_data, factory, term_idx=0)
    assert FPR_BOUNDS[0] <= rate <= FPR_BOUNDS[1], (
        f"FPR {rate:.1f}% outside {FPR_BOUNDS}"
    )


def test_high_complexity():
    """High spline count (n_splines=25), univariate noise."""
    factory = lambda: LinearGAM(s(0, n_splines=25), lam=0.6)
    rate = run_test_scenario(gen_noise_univariate, factory)
    assert FPR_BOUNDS[0] <= rate <= FPR_BOUNDS[1], (
        f"FPR {rate:.1f}% outside {FPR_BOUNDS}"
    )


def test_collinearity():
    """Two near-identical predictors. Convergence is allowed to fail; if it fits, FPR should be calibrated."""
    factory = lambda: LinearGAM(s(0) + s(1), lam=0.6)
    rate = run_test_scenario(gen_collinear_data, factory, term_idx=0)
    assert FPR_BOUNDS[0] <= rate <= FPR_BOUNDS[1], (
        f"FPR {rate:.1f}% outside {FPR_BOUNDS}"
    )
