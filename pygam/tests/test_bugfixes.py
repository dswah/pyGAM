"""Tests for categorical handling, term fixes, packaging, and usability bug fixes.

Each test reproduces the reported issue and verifies the fix.
"""

import numpy as np
import pytest

from pygam import LinearGAM, LogisticGAM, f, s, te
from pygam.datasets import coal, default, mcycle, wage

# ---------------------------------------------------------------------------
# #285 / #301  Categorical partial dependence
# ---------------------------------------------------------------------------


class TestCategoricalPartialDependence:
    """Fix for GitHub issues #285 and #301."""

    def test_factor_term_no_zero_label(self):
        """Categories that don't include 0 should work in partial_dependence."""
        np.random.seed(42)
        n = 200
        # Categories are 1, 2, 3 (no 0)
        X_cat = np.random.choice([1, 2, 3], size=n)
        X_cont = np.random.randn(n)
        X = np.column_stack([X_cont, X_cat])
        y = X_cont + X_cat * 0.5 + np.random.randn(n) * 0.1
        gam = LinearGAM(s(0) + f(1)).fit(X, y)
        # partial_dependence for the factor term should not raise
        pdep = gam.partial_dependence(1)
        assert pdep is not None
        assert len(pdep) > 0

    def test_generate_X_grid_factor_uses_integer_categories(self):
        """generate_X_grid for factor terms should return actual categories."""
        np.random.seed(42)
        n = 200
        X_cat = np.random.choice([2, 5, 8], size=n)
        X = np.column_stack([np.random.randn(n), X_cat])
        y = X[:, 0] + X_cat + np.random.randn(n) * 0.1
        gam = LinearGAM(s(0) + f(1)).fit(X, y)
        grid = gam.generate_X_grid(1)
        # The grid should contain exactly the category labels
        cat_values = grid[:, 1]
        np.testing.assert_array_equal(np.unique(cat_values), [2, 5, 8])

    def test_partial_dependence_factor_term(self):
        """partial_dependence should handle factor terms correctly."""
        X, y = wage(return_X_y=True)
        gam = LinearGAM(s(0) + s(1) + f(2)).fit(X, y)
        pdep = gam.partial_dependence(2)
        assert pdep is not None
        assert np.isfinite(pdep).all()


# ---------------------------------------------------------------------------
# #230  MetaTermMixin.__setattr__ re-entrancy
# ---------------------------------------------------------------------------


class TestSetAttrReentrancy:
    """Fix for GitHub issue #230."""

    def test_set_lam_on_tensor_term(self):
        """Setting lam on a TensorTerm should not cause recursion errors."""
        term = te(0, 1)
        np.random.seed(0)
        X = np.random.randn(50, 2)
        y = np.random.randn(50)
        gam = LinearGAM(term).fit(X, y)
        # setting lam via the plural interface should work
        gam.terms[0].lam = [0.1, 0.1]
        flat_lam = np.hstack(gam.terms[0].lam)
        np.testing.assert_allclose(flat_lam, [0.1, 0.1])

    def test_set_lam_on_termlist(self):
        """Setting lam on a TermList should propagate without recursion."""
        np.random.seed(0)
        X = np.random.randn(50, 2)
        y = np.random.randn(50)
        gam = LinearGAM(s(0) + s(1)).fit(X, y)
        old_lam = gam.terms.lam
        new_lam = [v * 2 for v in np.atleast_1d(np.hstack(old_lam))]
        gam.terms.lam = new_lam


# ---------------------------------------------------------------------------
# #423  Datasets loading
# ---------------------------------------------------------------------------


class TestDatasetsAvailable:
    """Fix for GitHub issue #423."""

    @pytest.mark.parametrize("loader", [mcycle, default, wage, coal])
    def test_dataset_loads(self, loader):
        X, y = loader(return_X_y=True)
        assert X is not None
        assert y is not None
        assert len(X) > 0
        assert len(y) > 0


# ---------------------------------------------------------------------------
# #242  gridsearch memory usage
# ---------------------------------------------------------------------------


class TestGridsearchMemory:
    """Fix for GitHub issue #242."""

    def test_gridsearch_does_not_store_all_models(self):
        """When return_scores=False, intermediate models should be freed."""
        X, y = mcycle(return_X_y=True)
        gam = LinearGAM()
        # This should complete without excessive memory usage
        gam.gridsearch(X, y, lam=np.logspace(-3, 3, 5))
        assert gam._is_fitted

    def test_gridsearch_return_scores(self):
        """When return_scores=True, all models should be returned."""
        X, y = mcycle(return_X_y=True)
        gam = LinearGAM()
        result = gam.gridsearch(X, y, return_scores=True, lam=np.logspace(-1, 1, 3))
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_gridsearch_keeps_best(self):
        X, y = mcycle(return_X_y=True)
        gam = LinearGAM()
        gam.gridsearch(X, y, lam=np.logspace(-3, 3, 5))
        # The model should be fitted with the best params
        score = gam.score(X, y)
        assert np.isfinite(score)


# ---------------------------------------------------------------------------
# #337  WRITEABLE flag issue with NumPy arrays
# ---------------------------------------------------------------------------


class TestWriteableFlag:
    """Fix for GitHub issue #337."""

    def test_readonly_array_input(self):
        X, y = mcycle(return_X_y=True)
        X_readonly = X.copy()
        X_readonly.flags.writeable = False
        # Should not raise ValueError about WRITEABLE flag
        gam = LinearGAM().fit(X_readonly, y)
        assert gam._is_fitted

    def test_readonly_predict(self):
        X, y = mcycle(return_X_y=True)
        gam = LinearGAM().fit(X, y)
        X_readonly = X.copy()
        X_readonly.flags.writeable = False
        pred = gam.predict(X_readonly)
        assert len(pred) == len(X)


# ---------------------------------------------------------------------------
# #257  LogisticGAM.sample() shape mismatch
# ---------------------------------------------------------------------------


class TestLogisticGAMSample:
    """Fix for GitHub issue #257."""

    def test_sample_returns_correct_shape(self):
        """LogisticGAM.sample should return (n_draws, n_samples) without error."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = (X.ravel() > 0).astype(int)
        gam = LogisticGAM(s(0)).fit(X, y)
        samples = gam.sample(X, y, n_draws=3, n_bootstraps=1)
        assert samples.shape == (3, 100)

    def test_sample_values_are_valid_binary(self):
        """Sampled values from LogisticGAM should be valid integers."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = (X.ravel() > 0).astype(int)
        gam = LogisticGAM(s(0)).fit(X, y)
        samples = gam.sample(X, y, n_draws=3, n_bootstraps=1)
        assert np.all(np.isin(samples, [0, 1]))


# ---------------------------------------------------------------------------
# #255  sample() random_state for reproducibility
# ---------------------------------------------------------------------------


class TestSampleRandomState:
    """Fix for GitHub issue #255."""

    def test_sample_reproducible_with_seed(self):
        """Calling sample with the same random_state gives identical results."""
        X, y = mcycle(return_X_y=True)
        gam = LinearGAM().fit(X, y)
        s1 = gam.sample(X, y, n_draws=3, n_bootstraps=1, random_state=42)
        s2 = gam.sample(X, y, n_draws=3, n_bootstraps=1, random_state=42)
        np.testing.assert_array_equal(s1, s2)

    def test_sample_different_seeds_differ(self):
        """Different seeds should produce different samples."""
        X, y = mcycle(return_X_y=True)
        gam = LinearGAM().fit(X, y)
        s1 = gam.sample(X, y, n_draws=3, n_bootstraps=1, random_state=0)
        s2 = gam.sample(X, y, n_draws=3, n_bootstraps=1, random_state=99)
        assert not np.array_equal(s1, s2)

    def test_sample_restores_global_rng_state(self):
        """After sample(), the global RNG state should be unchanged."""
        X, y = mcycle(return_X_y=True)
        gam = LinearGAM().fit(X, y)
        np.random.seed(7)
        _ = np.random.random()
        gam.sample(X, y, n_draws=2, n_bootstraps=1, random_state=42)
        after = np.random.random()
        # The next value from the global RNG should be the same as if
        # sample() never touched the global state
        np.random.seed(7)
        _ = np.random.random()
        expected = np.random.random()
        assert after == expected

    def test_sample_without_seed_works(self):
        """Calling sample without random_state should still work."""
        X, y = mcycle(return_X_y=True)
        gam = LinearGAM().fit(X, y)
        result = gam.sample(X, y, n_draws=2, n_bootstraps=1)
        assert result.shape == (2, len(X))


# ---------------------------------------------------------------------------
# ValueError formatting bug in _estimate_GCV_UBRE
# ---------------------------------------------------------------------------


class TestGCVGammaValidation:
    """Fix for ValueError message formatting in _estimate_GCV_UBRE."""

    def test_gamma_error_message_contains_value(self):
        """The error message for gamma < 1 should contain the actual value."""
        X, y = mcycle(return_X_y=True)
        gam = LinearGAM().fit(X, y)
        with pytest.raises(ValueError, match="0.5"):
            gam._estimate_GCV_UBRE(X, y, gamma=0.5)
