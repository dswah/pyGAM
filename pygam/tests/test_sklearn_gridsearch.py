import numpy as np
from sklearn.model_selection import GridSearchCV

from pygam import LogisticGAM


def test_logisticgam_sklearn_gridsearchcv():
    """
    Test that LogisticGAM can be used with sklearn's GridSearchCV.
    This protects against regressions related to Issue #482, where
    LogisticGAM lacked the `classes_` attribute and its `predict()`
    method did not return labels properly.
    """
    # Create simple binary classification data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    # y is intentionally given integer classes instead of boolean to
    # verify predict() un-maps them correctly
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, 0)

    # Note that LogisticGAM gridsearch internally natively, but
    # users often pipeline with Scikit-Learn's GridSearchCV
    param_grid = {"max_iter": [10, 100], "fit_intercept": [True, False]}

    gam = LogisticGAM()

    # This will fail with `NotFittedError` and `TypeError` unless
    # all our estimator compliance work is intact!
    grid = GridSearchCV(gam, param_grid, cv=3)
    grid.fit(X, y)

    # Ensure best estimator works as expected
    best_gam = grid.best_estimator_

    # Ensure classes_ was attached
    assert hasattr(best_gam, "classes_")
    assert np.array_equal(best_gam.classes_, np.array([0, 1]))

    # Ensure predict returns labels (not boolean probas)
    preds = best_gam.predict(X)
    assert preds.dtype.kind in "iu"  # Integer or Unsigned integer
    assert set(np.unique(preds)).issubset({0, 1})

    assert best_gam.score(X, y) >= 0.0
