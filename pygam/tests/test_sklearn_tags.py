import pytest

pytest.importorskip("sklearn")


def test_sklearn_tags_exists():
    from pygam import LinearGAM, LogisticGAM

    assert hasattr(LinearGAM(), "__sklearn_tags__")
    assert hasattr(LogisticGAM(), "__sklearn_tags__")


def test_check_estimator_runs():
    from sklearn.utils.estimator_checks import check_estimator

    from pygam import LinearGAM

    # Ensure sklearn compatibility checks run
    check_estimator(LinearGAM())


def test_logistic_gam_classifier_tag():
    from pygam import LogisticGAM

    gam = LogisticGAM()
    tags = gam.__sklearn_tags__()

    if not isinstance(tags, dict):
        assert tags.estimator_type == "classifier"


def test_linear_gam_regressor_tag():
    from pygam import LinearGAM

    gam = LinearGAM()
    tags = gam.__sklearn_tags__()

    if not isinstance(tags, dict):
        assert tags.estimator_type == "regressor"


def test_pipeline_compatibility():
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from pygam import LinearGAM

    X = np.random.rand(100, 3)
    y = np.random.rand(100)

    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("gam", LinearGAM()),
        ]
    )

    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert preds.shape == (100,)
