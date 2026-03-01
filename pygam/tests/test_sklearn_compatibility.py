"""Tests for scikit-learn API compatibility."""

import numpy as np

sklearn = __import__("pytest").importorskip("sklearn")

from pygam import LinearGAM


def test_sklearn_tags():
    """LinearGAM should expose __sklearn_tags__ from BaseEstimator."""
    gam = LinearGAM()
    assert hasattr(gam, "__sklearn_tags__")


def test_n_features_in_set_after_fit():
    """n_features_in_ should be set after fitting."""
    X = np.random.rand(50, 3)
    y = np.random.rand(50)
    gam = LinearGAM().fit(X, y)
    assert hasattr(gam, "n_features_in_")
    assert gam.n_features_in_ == 3


def test_not_fitted_error():
    """Calling predict on unfitted model should raise NotFittedError."""
    from sklearn.exceptions import NotFittedError

    gam = LinearGAM()
    try:
        gam.predict(np.random.rand(10, 3))
        assert False, "Should have raised NotFittedError"
    except NotFittedError:
        pass


def test_sparse_input_rejected():
    """Sparse input should be explicitly rejected with TypeError."""
    from scipy.sparse import csr_array

    gam = LinearGAM()
    X_sparse = csr_array(np.random.rand(50, 3))
    y = np.random.rand(50)
    try:
        gam.fit(X_sparse, y)
        assert False, "Should have raised TypeError for sparse input"
    except TypeError as e:
        assert "Sparse" in str(e)


def test_complex_input_rejected():
    """Complex input should be explicitly rejected with ValueError."""
    gam = LinearGAM()
    X_complex = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
    y = np.array([1, 2])
    try:
        gam.fit(X_complex, y)
        assert False, "Should have raised ValueError for complex input"
    except ValueError as e:
        assert "Complex data not supported" in str(e)


def test_callbacks_default_is_tuple():
    """callbacks default should be a tuple, not a mutable list."""
    gam = LinearGAM()
    assert isinstance(gam.callbacks, tuple)


def test_callbacks_preserved_after_fit():
    """callbacks param should be unchanged after fit()."""
    gam = LinearGAM()
    original_callbacks = gam.callbacks
    X = np.random.rand(50, 3)
    y = np.random.rand(50)
    gam.fit(X, y)
    assert gam.callbacks == original_callbacks
    assert isinstance(gam.callbacks, tuple)


def test_zero_features_rejected():
    """Input with 0 features should be rejected."""
    gam = LinearGAM()
    X_empty = np.empty(0).reshape(10, 0)
    y = np.ones(10)
    try:
        gam.fit(X_empty, y)
        assert False, "Should have raised ValueError for 0 features"
    except ValueError as e:
        assert "0 feature" in str(e)
