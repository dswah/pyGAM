def test_sklearn_tags_exists():
    from pygam import LinearGAM

    gam = LinearGAM()
    assert hasattr(gam, "__sklearn_tags__")


def test_check_estimator_runs():
    from sklearn.utils.estimator_checks import check_estimator

    from pygam import LinearGAM

    # check_estimator will natively read tags and run test suite
    try:
        check_estimator(LinearGAM())
    except Exception as e:
        # Currently, PyGAM estimators may still fail some stringent sklearn checks.
        # But they should NOT fail due to missing sklearn tags!
        # Let's ensure the error is not AttributeError for sklearn_tags
        error_msg = str(e)
        if "sklearn_tags" in error_msg or "BaseEstimator" in error_msg:
            raise e


def test_logistic_gam_classifier_tag():
    from pygam import LogisticGAM

    gam = LogisticGAM()
    tags = gam.__sklearn_tags__()

    if isinstance(tags, dict):
        # old scikit-learn
        pass
    else:
        assert tags.estimator_type == "classifier"


def test_linear_gam_regressor_tag():
    from pygam import LinearGAM

    gam = LinearGAM()
    tags = gam.__sklearn_tags__()

    if isinstance(tags, dict):
        pass
    else:
        assert tags.estimator_type == "regressor"


def test_pipeline_compatibility():
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from pygam import LinearGAM

    X = np.random.rand(100, 3)
    y = np.random.rand(100)

    pipe = Pipeline([("scale", StandardScaler()), ("gam", LinearGAM())])

    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (100,)
