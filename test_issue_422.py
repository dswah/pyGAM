"""Test script to verify fix for issue #422 - sklearn 1.7+ compatibility."""

import numpy as np
from pygam import GAM
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import KFold, RandomizedSearchCV

print("Testing sklearn 1.7+ compatibility with pyGAM...")
print(f"sklearn version: {__import__('sklearn').__version__}")
print(f"pygam version: {__import__('pygam').__version__}")

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 3)
y = X[:, 0] + 0.5 * X[:, 1] ** 2 + np.random.randn(100) * 0.1

# Test the exact code from the issue
scorer = make_scorer(r2_score, greater_is_better=True)
random_search = RandomizedSearchCV(
    GAM(),
    cv=KFold(n_splits=3),
    param_distributions={"n_splines": np.arange(5, 40)},
    n_iter=20,
    scoring=scorer,
    verbose=0,
    return_train_score=True,
)

print("\nFitting RandomizedSearchCV with GAM...")
random_search.fit(X, y)

print(f"✓ Success! Best score: {random_search.best_score_:.4f}")
print(f"✓ Best params: {random_search.best_params_}")
print(f"✓ GAM has __sklearn_tags__: {hasattr(random_search.best_estimator_, '__sklearn_tags__')}")
print("\nIssue #422 is FIXED!")
