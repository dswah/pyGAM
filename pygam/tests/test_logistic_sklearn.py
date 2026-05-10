import numpy as np
import pytest

from pygam import LogisticGAM


def test_logistic_string_labels():
    """Test that LogisticGAM natively handles string labels and exposes classes_"""
    np.random.seed(42)
    X = np.random.rand(10, 2)
    y_str = np.array(
        ["spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham"]
    )

    gam = LogisticGAM().fit(X, y_str)
    preds = gam.predict(X)

    assert hasattr(gam, "classes_"), "LogisticGAM did not expose classes_ attribute"
    assert list(gam.classes_) == ["ham", "spam"], "Classes were not encoded correctly"
    assert preds.dtype.kind in {"U", "S", "O"}, "Predictions did not return as strings"


def test_logistic_legacy_boolean_labels():
    """Test that LogisticGAM still perfectly handles legacy boolean arrays"""
    np.random.seed(42)
    X = np.random.rand(10, 2)
    y_bool = X[:, 0] > 0.5

    gam = LogisticGAM().fit(X, y_bool)
    preds = gam.predict(X)

    assert hasattr(gam, "classes_")
    assert preds.dtype == bool or preds.dtype == np.bool_


def test_logistic_multiclass_rejection():
    """Test that LogisticGAM strictly rejects multi-class data (only supports binary)"""
    X = np.random.rand(6, 2)
    y_multi = np.array([0, 1, 2, 0, 1, 2])

    with pytest.raises(
        ValueError, match="LogisticGAM requires binary classification data"
    ):
        LogisticGAM().fit(X, y_multi)
