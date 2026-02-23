"""Tests for Wood (2013b) p-value computation in GAM.

Wood, S.N. (2013b). On p-values for smooth components of an extended
generalized additive model. Biometrika, 100(1), 221-228.
doi:10.1093/biomet/ass048
"""

import numpy as np
import pytest

from pygam import LinearGAM, PoissonGAM, f, s

N = 500


@pytest.fixture(scope="module")
def linear_data():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 6, N)
    y = np.sin(x) + rng.normal(0, 0.3, N)
    return x.reshape(-1, 1), y


@pytest.fixture(scope="module")
def linear_gam_fit(linear_data):
    X, y = linear_data
    return LinearGAM(s(0)).fit(X, y)


@pytest.fixture(scope="module")
def gam_with_noise(linear_data):
    X, y = linear_data
    rng = np.random.default_rng(7)
    noise_col = rng.standard_normal(N)
    X_ext = np.c_[X, noise_col]
    return LinearGAM(s(0) + s(1)).fit(X_ext, y)


class TestPValueRange:
    def test_all_in_unit_interval(self, linear_gam_fit):
        pvals = linear_gam_fit._estimate_p_values()
        assert all(0.0 <= p <= 1.0 for p in pvals)

    def test_returns_one_per_term(self, linear_gam_fit):
        pvals = linear_gam_fit._estimate_p_values()
        assert len(pvals) == len(linear_gam_fit.terms)


class TestSignificance:
    def test_signal_term_is_significant(self, linear_gam_fit):
        """A spline fitted on sin(x) should have p < 0.01."""
        pvals = linear_gam_fit._estimate_p_values()
        assert pvals[0] < 0.01

    def test_noise_term_is_not_significant(self, gam_with_noise):
        """A pure-noise feature should not be significant at α=0.01."""
        pvals = gam_with_noise._estimate_p_values()
        # s(1) is the noise column; last term is intercept
        assert pvals[1] > 0.01

    def test_signal_term_survives_with_noise_column(self, gam_with_noise):
        """Adding noise should not destroy significance of the real term."""
        pvals = gam_with_noise._estimate_p_values()
        assert pvals[0] < 0.01


class TestDistributionDispatch:
    def test_known_scale_uses_chi2_path(self, linear_data):
        """PoissonGAM has known scale → chi2 branch."""
        X, y = linear_data
        y_count = np.floor(np.abs(y) * 3).astype(int) + 1
        gam = PoissonGAM(s(0)).fit(X, y_count)
        pvals = gam._estimate_p_values()
        assert all(0.0 <= p <= 1.0 for p in pvals)
        assert pvals[0] < 0.01  # sin signal still significant

    def test_unknown_scale_uses_f_path(self, linear_gam_fit):
        """LinearGAM estimates scale → F-distribution branch."""
        assert not linear_gam_fit.distribution._known_scale
        pvals = linear_gam_fit._estimate_p_values()
        assert all(0.0 <= p <= 1.0 for p in pvals)


class TestEdgeCases:
    def test_over_parameterized_no_crash(self, linear_data):
        """n_splines > n_samples must not raise IndexError."""
        X, y = linear_data
        X_small, y_small = X[:30], y[:30]
        gam = LinearGAM(s(0, n_splines=50)).fit(X_small, y_small)
        pvals = gam._estimate_p_values()
        assert all(0.0 <= p <= 1.0 for p in pvals)

    def test_factor_term_pvalue(self):
        """Factor terms should produce valid p-values."""
        rng = np.random.default_rng(1)
        X = rng.integers(0, 5, size=(200, 1))
        y = X[:, 0].astype(float) * 2.0 + rng.normal(0, 0.5, 200)
        gam = LinearGAM(f(0)).fit(X, y)
        pvals = gam._estimate_p_values()
        assert all(0.0 <= p <= 1.0 for p in pvals)
        assert pvals[0] < 0.01  # factor-encoded feature is informative
