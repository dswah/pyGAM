import numpy as np
import pytest

from pygam import (
    GAM,
    ExpectileGAM,
    GammaGAM,
    InvGaussGAM,
    LinearGAM,
    LogisticGAM,
    PoissonGAM,
)


def test_can_build_sub_models():
    """
    check that the inits of all the sub-models are correct
    """
    LinearGAM()
    LogisticGAM()
    PoissonGAM()
    GammaGAM()
    InvGaussGAM()
    ExpectileGAM()
    assert True


def test_LinearGAM_uni(mcycle_X_y):
    """
    check that we can fit a Linear GAM on real, univariate data
    """
    X, y = mcycle_X_y
    gam = LinearGAM().fit(X, y)
    assert gam._is_fitted


def test_LinearGAM_multi(wage_X_y):
    """
    check that we can fit a Linear GAM on real, multivariate data
    """
    X, y = wage_X_y
    gam = LinearGAM().fit(X, y)
    assert gam._is_fitted


def test_LogisticGAM(default_X_y):
    """
    check that we can fit a Logistic GAM on real data
    """
    X, y = default_X_y
    gam = LogisticGAM().fit(X, y)
    assert gam._is_fitted


def test_LogisticGAM_numeric_targets_unchanged(default_X_y):
    """
    check that 0/1 targets keep the existing behaviour and expose classes_
    """
    X, y = default_X_y
    gam = LogisticGAM().fit(X, y)
    np.testing.assert_array_equal(gam.classes_, [0, 1])
    assert gam.predict(X).dtype == bool


@pytest.mark.parametrize("labels", [("no", "yes"), (-1, 1), (3, 7)])
def test_LogisticGAM_class_labels(default_X_y, labels):
    """
    check that LogisticGAM accepts any two class labels, see #578
    """
    X, y = default_X_y
    y_labels = np.where(y == 1, labels[1], labels[0])

    gam_num = LogisticGAM().fit(X, y)
    gam = LogisticGAM().fit(X, y_labels)

    np.testing.assert_array_equal(gam.classes_, labels)
    np.testing.assert_allclose(gam.predict_proba(X), gam_num.predict_proba(X))

    y_pred = gam.predict(X)
    assert set(np.unique(y_pred)) <= set(labels)
    np.testing.assert_array_equal(y_pred == labels[1], gam_num.predict(X))

    assert gam.score(X, y_labels) == pytest.approx(gam_num.score(X, y))
    np.testing.assert_allclose(
        gam.loglikelihood(X, y_labels), gam_num.loglikelihood(X, y)
    )
    np.testing.assert_allclose(
        gam.deviance_residuals(X, y_labels), gam_num.deviance_residuals(X, y)
    )


def test_LogisticGAM_rejects_wrong_number_of_classes(default_X_y):
    """
    check that LogisticGAM raises a clear error unless there are 2 classes
    """
    X, y = default_X_y
    with pytest.raises(ValueError, match="exactly 2 classes"):
        LogisticGAM().fit(X, np.array(["a", "b", "c"])[np.arange(len(y)) % 3])
    with pytest.raises(ValueError, match="exactly 2 classes"):
        LogisticGAM().fit(X, np.full(len(y), "a"))


def test_LogisticGAM_rejects_unseen_labels(default_X_y):
    """
    check that scoring with labels not seen in fit raises a clear error
    """
    X, y = default_X_y
    gam = LogisticGAM().fit(X, np.where(y == 1, "yes", "no"))
    with pytest.raises(ValueError, match="not seen during fit"):
        gam.score(X, np.where(y == 1, "yes", "maybe"))


def test_LogisticGAM_gridsearch_class_labels(default_X_y):
    """
    check that gridsearch accepts class labels and keeps classes_
    """
    X, y = default_X_y
    y_labels = np.where(y == 1, "yes", "no")
    gam = LogisticGAM().gridsearch(
        X, y_labels, lam=np.logspace(-1, 1, 2), progress=False
    )
    np.testing.assert_array_equal(gam.classes_, ["no", "yes"])
    assert set(np.unique(gam.predict(X))) <= {"no", "yes"}


def test_PoissonGAM(coal_X_y):
    """
    check that we can fit a Poisson GAM on real data
    """
    X, y = coal_X_y
    gam = PoissonGAM().fit(X, y)
    assert gam._is_fitted


def test_InvGaussGAM(trees_X_y):
    """
    check that we can fit a InvGauss GAM on real data
    """
    X, y = trees_X_y
    gam = InvGaussGAM().fit(X, y)
    assert gam._is_fitted


def test_GammaGAM(trees_X_y):
    """
    check that we can fit a Gamma GAM on real data
    """
    X, y = trees_X_y
    gam = GammaGAM().fit(X, y)
    assert gam._is_fitted


def test_CustomGAM(trees_X_y):
    """
    check that we can fit a Custom GAM on real data
    """
    X, y = trees_X_y
    gam = GAM(distribution="gamma", link="inverse").fit(X, y)
    assert gam._is_fitted


def test_ExpectileGAM_uni(mcycle_X_y):
    """
    check that we can fit an Expectile GAM on real, univariate data
    """
    X, y = mcycle_X_y
    gam = ExpectileGAM().fit(X, y)
    assert gam._is_fitted


def test_ExpectileGAM_bad_expectiles(mcycle_X_y):
    """
    check that get errors for unacceptable expectiles
    """
    X, y = mcycle_X_y
    with pytest.raises(ValueError):
        ExpectileGAM(expectile=0).fit(X, y)
    with pytest.raises(ValueError):
        ExpectileGAM(expectile=1).fit(X, y)
    with pytest.raises(ValueError):
        ExpectileGAM(expectile=-0.1).fit(X, y)
    with pytest.raises(ValueError):
        ExpectileGAM(expectile=1.1).fit(X, y)


# TODO check dicts: DISTRIBUTIONS etc
