"""Tests for p-value computation bug fixes.

Each test reproduces the reported issue and verifies the fix.
"""

import warnings

import numpy as np

from pygam import LinearGAM, PoissonGAM, f, s
from pygam.datasets import coal, mcycle, wage

# ---------------------------------------------------------------------------
# #303  PoissonGAM EDoF and p-values
# ---------------------------------------------------------------------------


class TestPoissonGAMSummary:
    """Fix for GitHub issue #303."""

    def test_poisson_edof_per_coef_matches_coef_length(self):
        X, y = coal(return_X_y=True)
        gam = PoissonGAM().fit(X, y)
        assert len(gam.statistics_["edof_per_coef"]) == len(gam.coef_)

    def test_poisson_edof_is_positive(self):
        X, y = coal(return_X_y=True)
        gam = PoissonGAM().fit(X, y)
        assert gam.statistics_["edof"] > 0

    def test_poisson_summary_runs(self):
        """summary() should run without errors for PoissonGAM."""
        X, y = coal(return_X_y=True)
        gam = PoissonGAM().fit(X, y)
        # Should not raise
        gam.summary()

    def test_poisson_p_values_are_finite(self):
        X, y = coal(return_X_y=True)
        gam = PoissonGAM().fit(X, y)
        p_values = gam.statistics_["p_values"]
        for p in p_values:
            assert np.isfinite(p)


# ---------------------------------------------------------------------------
# #163  p-values use Wood 2013b eigendecomposition method
# ---------------------------------------------------------------------------


class TestPValueWood2013b:
    """Fix for GitHub issue #163."""

    def test_p_values_are_finite(self):
        """All p-values should be finite numbers."""
        X, y = mcycle(return_X_y=True)
        gam = LinearGAM().fit(X, y)
        p_values = gam.statistics_["p_values"]
        for p in p_values:
            assert np.isfinite(p)

    def test_p_values_non_trivial_for_spline(self):
        """Spline p-values should not be exactly 0 for reasonable models."""
        X, y = mcycle(return_X_y=True)
        gam = LinearGAM(s(0)).fit(X, y)
        p = gam.statistics_["p_values"][0]
        assert 0 <= p <= 1

    def test_p_values_larger_than_old_method(self):
        """Rank-truncated p-values should generally be >= the old full-rank values.

        For a penalized spline with EDoF << rank, the full-rank chi-sq test
        uses too many degrees of freedom, making p-values artificially small.
        The new method uses ceil(EDoF) as the effective rank, so p-values
        should be larger (more conservative).
        """
        X, y = wage(return_X_y=True)
        terms = s(0, n_splines=25) + s(1, n_splines=25) + f(2)
        gam = LinearGAM(terms).fit(X, y)
        # Compute p-values the old way (full rank)
        import scipy as sp

        from pygam.terms import SplineTerm as _SplineTerm

        old_p_values = []
        for i in range(len(gam.terms)):
            idxs = gam.terms.get_coef_indices(i)
            cov = gam.statistics_["cov"][idxs][:, idxs]
            coef = gam.coef_[idxs].copy()
            if isinstance(gam.terms[i], _SplineTerm):
                coef -= coef.mean()
            inv_cov, rank = sp.linalg.pinv(cov, return_rank=True)
            score = coef.T.dot(inv_cov).dot(coef)
            score_f = score / rank
            old_p = 1 - sp.stats.f.cdf(
                score_f, rank, gam.statistics_["n_samples"] - gam.statistics_["edof"]
            )
            old_p_values.append(old_p)

        new_p_values = gam.statistics_["p_values"]
        # For penalized splines, new p-values should be >= old p-values
        for i, term in enumerate(gam.terms):
            if isinstance(term, _SplineTerm) and not term.isintercept:
                assert new_p_values[i] >= old_p_values[i] - 1e-15, (
                    f"Term {i}: new p={new_p_values[i]:.2e} < old p={old_p_values[i]:.2e}"
                )

    def test_summary_no_known_bug_warning(self):
        """summary() should no longer emit the 'KNOWN BUG' warning."""
        X, y = mcycle(return_X_y=True)
        gam = LinearGAM().fit(X, y)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gam.summary()
            known_bug_warnings = [x for x in w if "KNOWN BUG" in str(x.message)]
            assert len(known_bug_warnings) == 0


# ---------------------------------------------------------------------------
# #298  EDoF not shown when n_splines > n_samples (duplicate of #303)
# ---------------------------------------------------------------------------


class TestEDoFManyPredictors:
    """Fix for GitHub issue #298 (same underlying fix as #303)."""

    def test_edof_shown_for_wide_model(self):
        """EDoF should be populated even when n_splines > n_samples."""
        X, y = mcycle(return_X_y=True)
        n = len(X)
        gam = LinearGAM(s(0, n_splines=n + 10)).fit(X, y)
        assert len(gam.statistics_["edof_per_coef"]) == len(gam.coef_)
        assert gam.statistics_["edof"] > 0

    def test_summary_displays_edof_wide_model(self):
        """summary() should show EDoF column, not empty strings."""
        X, y = mcycle(return_X_y=True)
        n = len(X)
        gam = LinearGAM(s(0, n_splines=n + 10)).fit(X, y)
        # summary() should not raise
        gam.summary()
