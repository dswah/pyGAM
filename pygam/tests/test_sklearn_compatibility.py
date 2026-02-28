import pytest
import numpy as np
from pygam import GAM
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, make_scorer

def test_sklearn_compatibility_randomized_search():
    """
    Test that pyGAM models can be used with Scikit-Learn's RandomizedSearchCV.
    This explicitly tests for the presence of Scikit-Learn 1.6+ __sklearn_tags__
    support through BaseEstimator inheritance.
    """
    X = np.random.rand(50, 2)
    y = np.random.rand(50)

    scorer = make_scorer(r2_score, greater_is_better=True)
    random_search = RandomizedSearchCV(
        GAM(),
        cv=KFold(n_splits=2),
        param_distributions={"n_splines": [5, 10]},
        n_iter=1,
        scoring=scorer,
        verbose=0,
    )
    
    # This will raise an AttributeError if __sklearn_tags__ or BaseEstimator is missing
    random_search.fit(X, y)
    
    assert hasattr(random_search, "best_estimator_")
