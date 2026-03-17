"""Tests for scikit-learn compatibility (v1.7+)."""

import numpy as np
import pytest

from pygam import GAM, LinearGAM, LogisticGAM, PoissonGAM, GammaGAM, InvGaussGAM, ExpectileGAM

# Check if sklearn is available
try:
    from sklearn.base import BaseEstimator
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
    from sklearn.metrics import r2_score, make_scorer, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestSklearnCompatibility:
    """Test suite for scikit-learn compatibility."""

    def test_gam_has_sklearn_tags(self):
        """Test that GAM has __sklearn_tags__ method."""
        gam = GAM()
        assert hasattr(gam, '__sklearn_tags__')
        tags = gam.__sklearn_tags__()
        assert tags is not None

    def test_linear_gam_has_sklearn_tags(self):
        """Test that LinearGAM has __sklearn_tags__ method."""
        gam = LinearGAM()
        assert hasattr(gam, '__sklearn_tags__')
        tags = gam.__sklearn_tags__()
        assert tags is not None

    def test_logistic_gam_has_sklearn_tags(self):
        """Test that LogisticGAM has __sklearn_tags__ method."""
        gam = LogisticGAM()
        assert hasattr(gam, '__sklearn_tags__')
        tags = gam.__sklearn_tags__()
        assert tags is not None

    def test_gam_inherits_from_base_estimator(self):
        """Test that GAM inherits from BaseEstimator."""
        gam = GAM()
        assert isinstance(gam, BaseEstimator)

    def test_linear_gam_with_randomized_search_cv(self):
        """Test LinearGAM with RandomizedSearchCV (issue #422)."""
        # Generate sample data
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] + 0.5 * X[:, 1] ** 2 + np.random.randn(100) * 0.1

        # Create scorer
        scorer = make_scorer(r2_score, greater_is_better=True)

        # Create RandomizedSearchCV
        random_search = RandomizedSearchCV(
            LinearGAM(),
            cv=KFold(n_splits=3),
            param_distributions={"n_splines": np.arange(5, 15)},
            n_iter=3,
            scoring=scorer,
            verbose=0,
            return_train_score=True,
        )

        # This should not raise AttributeError about __sklearn_tags__
        random_search.fit(X, y)

        # Verify the search completed
        assert hasattr(random_search, 'best_estimator_')
        assert hasattr(random_search, 'best_score_')

    def test_logistic_gam_with_grid_search_cv(self):
        """Test LogisticGAM with GridSearchCV."""
        # Generate sample data
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # Create scorer
        scorer = make_scorer(accuracy_score, greater_is_better=True)

        # Create GridSearchCV
        grid_search = GridSearchCV(
            LogisticGAM(),
            cv=KFold(n_splits=3),
            param_grid={"n_splines": [5, 10]},
            scoring=scorer,
            verbose=0,
        )

        # This should not raise AttributeError about __sklearn_tags__
        grid_search.fit(X, y)

        # Verify the search completed
        assert hasattr(grid_search, 'best_estimator_')
        assert hasattr(grid_search, 'best_score_')

    def test_poisson_gam_with_randomized_search_cv(self):
        """Test PoissonGAM with RandomizedSearchCV."""
        # Generate sample data
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = np.random.poisson(np.exp(X[:, 0] * 0.5))

        # Create RandomizedSearchCV
        random_search = RandomizedSearchCV(
            PoissonGAM(),
            cv=KFold(n_splits=3),
            param_distributions={"n_splines": np.arange(5, 15)},
            n_iter=3,
            verbose=0,
        )

        # This should not raise AttributeError about __sklearn_tags__
        random_search.fit(X, y)

        # Verify the search completed
        assert hasattr(random_search, 'best_estimator_')

    def test_all_gam_types_have_tags(self):
        """Test that all GAM types have __sklearn_tags__ method."""
        gam_classes = [GAM, LinearGAM, LogisticGAM, PoissonGAM, GammaGAM, InvGaussGAM, ExpectileGAM]
        
        for gam_class in gam_classes:
            gam = gam_class()
            assert hasattr(gam, '__sklearn_tags__'), f"{gam_class.__name__} missing __sklearn_tags__"
            tags = gam.__sklearn_tags__()
            assert tags is not None, f"{gam_class.__name__}.__sklearn_tags__() returned None"

    def test_gam_get_params(self):
        """Test that GAM.get_params() works correctly."""
        gam = LinearGAM(max_iter=200, tol=1e-5)
        params = gam.get_params()
        
        assert 'max_iter' in params
        assert params['max_iter'] == 200
        assert 'tol' in params
        assert params['tol'] == 1e-5

    def test_gam_set_params(self):
        """Test that GAM.set_params() works correctly."""
        gam = LinearGAM()
        gam.set_params(max_iter=200, tol=1e-5)
        
        assert gam.max_iter == 200
        assert gam.tol == 1e-5
