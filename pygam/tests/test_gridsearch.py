import numpy as np
import pandas as pd
import pytest

from pygam import (
    GammaGAM,
    InvGaussGAM,
    LinearGAM,
    LogisticGAM,
    PoissonGAM,
)


def test_gridsearch_returns_scores(mcycle_X_y):
    """
    check that gridsearch returns the expected number of models
    """
    n = 5
    X, y = mcycle_X_y

    gam = LinearGAM()
    scores = gam.gridsearch(X, y, lam=np.logspace(-3, 3, n), return_scores=True)

    assert len(scores) == n


def test_gridsearch_returns_extra_score_if_fitted(mcycle_X_y):
    """
    check that gridsearch returns an extra score if our model is pre-fitted
    """
    n = 5
    X, y = mcycle_X_y

    gam = LinearGAM().fit(X, y)
    scores = gam.gridsearch(X, y, lam=np.logspace(-3, 3, n), return_scores=True)

    assert len(scores) == n + 1


def test_gridsearch_keep_best(mcycle_X_y):
    """
    check that gridsearch returns worse model if keep_best=False
    """
    n = 5
    X, y = mcycle_X_y

    gam = LinearGAM(lam=1000000).fit(X, y)
    score1 = gam.statistics_["GCV"]

    scores = gam.gridsearch(
        X, y, lam=np.logspace(-3, 3, n), keep_best=False, return_scores=True
    )

    assert np.min(list(scores.values())) < score1


def test_gridsearch_improves_objective(mcycle_X_y):
    """
    check that gridsearch improves model objective
    """
    n = 21
    X, y = mcycle_X_y

    gam = LinearGAM().fit(X, y)
    objective_0 = gam.statistics_["GCV"]

    gam = LinearGAM().gridsearch(X, y, lam=np.logspace(-2, 0, n))
    objective_1 = gam.statistics_["GCV"]

    assert objective_1 <= objective_0


def test_gridsearch_all_dimensions_same(cake_X_y):
    """
    check that gridsearch searches all dimensions of lambda with equal values
    """
    n = 5
    X, y = cake_X_y

    scores = LinearGAM().gridsearch(X, y, lam=np.logspace(-3, 3, n), return_scores=True)

    assert len(scores) == n
    assert X.shape[1] > 1


def test_gridsearch_all_dimensions_independent(cake_X_y):
    """
    check that gridsearch searches all dimensions of lambda independently
    """
    n = 4
    X, y = cake_X_y
    m = X.shape[1]

    scores = LinearGAM().gridsearch(
        X, y, lam=[np.logspace(-3, 3, n)] * m, return_scores=True
    )

    assert len(scores) == n**m
    assert m > 1


def test_no_cartesian_product(cake_X_y):
    """
    check that gridsearch does not do a cartesian product when a 2D numpy array is
    passed as the grid and the number of columns matches the len of the parameter
    """
    n = 5
    X, y = cake_X_y
    m = X.shape[1]

    lams = np.array([np.logspace(-3, 3, n)] * m).T
    assert lams.shape == (n, m)

    scores = LinearGAM().gridsearch(X, y, lam=lams, return_scores=True)

    assert len(scores) == n
    assert m > 1


def test_wrong_grid_shape(cake_X_y):
    """
    check that gridsearch raises a ValueError when the grid shape cannot be interpreted
    """
    X, y = cake_X_y
    lams = np.random.rand(50, X.shape[1] + 1)

    with pytest.raises(ValueError):
        LinearGAM().gridsearch(X, y, lam=lams, return_scores=True)

    lams = lams.T.tolist()
    assert len(lams) == X.shape[1] + 1
    with pytest.raises(ValueError):
        LinearGAM().gridsearch(X, y, lam=lams, return_scores=True)


def test_GCV_objective_is_for_unknown_scale(
    mcycle_X_y, default_X_y, coal_X_y, trees_X_y
):
    """
    check that we use the GCV objective only for models with unknown scale

    &

    attempting to use it for models with known scale should return ValueError
    """
    lam = np.linspace(1e-3, 1e3, 2)

    unknown_scale = [
        (LinearGAM, mcycle_X_y),
        (GammaGAM, trees_X_y),
        (InvGaussGAM, trees_X_y),
    ]

    known_scale = [(LogisticGAM, default_X_y), (PoissonGAM, coal_X_y)]

    for gam, (X, y) in unknown_scale:
        scores1 = list(
            gam()
            .gridsearch(X, y, lam=lam, objective="auto", return_scores=True)
            .values()
        )
        scores2 = list(
            gam()
            .gridsearch(X, y, lam=lam, objective="GCV", return_scores=True)
            .values()
        )
        assert np.allclose(scores1, scores2)

    for gam, (X, y) in known_scale:
        try:
            list(
                gam()
                .gridsearch(X, y, lam=lam, objective="GCV", return_scores=True)
                .values()
            )
        except ValueError:
            assert True


def test_UBRE_objective_is_for_known_scale(
    mcycle_X_y, default_X_y, coal_X_y, trees_X_y
):
    """
    check that we use the UBRE objective only for models with known scale

    &

    attempting to use it for models with unknown scale should return ValueError
    """
    lam = np.linspace(1e-3, 1e3, 2)

    unknown_scale = [
        (LinearGAM, mcycle_X_y),
        (GammaGAM, trees_X_y),
        (InvGaussGAM, trees_X_y),
    ]

    known_scale = [(LogisticGAM, default_X_y), (PoissonGAM, coal_X_y)]

    for gam, (X, y) in known_scale:
        scores1 = list(
            gam()
            .gridsearch(X, y, lam=lam, objective="auto", return_scores=True)
            .values()
        )
        scores2 = list(
            gam()
            .gridsearch(X, y, lam=lam, objective="UBRE", return_scores=True)
            .values()
        )
        assert np.allclose(scores1, scores2)

    for gam, (X, y) in unknown_scale:
        try:
            list(
                gam()
                .gridsearch(X, y, lam=lam, objective="UBRE", return_scores=True)
                .values()
            )
        except ValueError:
            assert True


def test_no_models_fitted(mcycle_X_y):
    """
    test no models fitted returns original gam
    """
    X, y = mcycle_X_y
    scores = LinearGAM().gridsearch(X, y, lam=[-3, -2, -1], return_scores=True)

    # scores is not a dict of scores but an (unfitted) gam!
    assert not isinstance(scores, dict)
    assert isinstance(scores, LinearGAM)
    assert not scores._is_fitted


def test_param_validation_order_REGRESSION():
    """
    test order of operations in parameter validation for gridsearch

    we should be able to gridsearch on a 1-D X array

    the reason is that validation of data-dependent parameters should occur AFTER
    validation of data.
    """
    X = np.arange(10)
    y = X**2

    gam = LinearGAM().gridsearch(X, y)
    assert gam._is_fitted


def test_gridsearch_works_on_Series_REGRESSION():
    """
    we should be able to do a gridsearch() on a Pandas DataFrame and Series
    just like we can do fit()
    """

    X = pd.DataFrame(np.arange(100))
    y = X**2

    # DataFrame
    gam = LinearGAM().gridsearch(X, y)
    assert gam._is_fitted

    # Series
    gam = LinearGAM().gridsearch(X[0], y)
    assert gam._is_fitted


def test_gridsearch_gamma_favors_smoother_models(mcycle_X_y):
    """
    check that higher gamma in gridsearch favors smoother models.

    a larger gamma exaggerates the effective degrees of freedom penalty
    in GCV/UBRE, so the optimizer should prefer a larger lambda
    (more regularization = smoother).
    """
    X, y = mcycle_X_y
    lam_grid = np.logspace(-3, 3, 11)

    gam_default = LinearGAM().gridsearch(X, y, lam=lam_grid, gamma=1.4)
    gam_smooth = LinearGAM().gridsearch(X, y, lam=lam_grid, gamma=5.0)

    # higher gamma should select equal or higher lam
    from pygam.utils import flatten

    lam_default = np.sum(flatten(gam_default.lam))
    lam_smooth = np.sum(flatten(gam_smooth.lam))
    assert lam_smooth >= lam_default


def test_gridsearch_gamma_changes_scores(mcycle_X_y):
    """
    check that different gamma values produce different GCV scores
    for the same model.
    """
    X, y = mcycle_X_y
    lam_grid = np.logspace(-3, 3, 5)

    scores_default = LinearGAM().gridsearch(
        X, y, lam=lam_grid, gamma=1.4, return_scores=True
    )
    scores_high = LinearGAM().gridsearch(
        X, y, lam=lam_grid, gamma=3.0, return_scores=True
    )

    # scores should differ because gamma changes the GCV formula
    vals_default = list(scores_default.values())
    vals_high = list(scores_high.values())
    assert not np.allclose(vals_default, vals_high)


def test_gridsearch_random_sampling(cake_X_y):
    """
    check that n_random_samples limits the number of candidate models.
    """
    n = 4
    X, y = cake_X_y
    m = X.shape[1]

    # without random sampling: n^m combinations
    scores_full = LinearGAM().gridsearch(
        X, y, lam=[np.logspace(-3, 3, n)] * m, return_scores=True
    )
    assert len(scores_full) == n**m

    # with random sampling: only n_random_samples combinations
    n_samples = 5
    scores_random = LinearGAM().gridsearch(
        X, y, lam=[np.logspace(-3, 3, n)] * m,
        n_random_samples=n_samples, return_scores=True
    )
    assert len(scores_random) == n_samples


def test_gridsearch_random_sampling_larger_than_grid(mcycle_X_y):
    """
    if n_random_samples >= grid size, all candidates are tested.
    """
    n = 5
    X, y = mcycle_X_y

    scores = LinearGAM().gridsearch(
        X, y, lam=np.logspace(-3, 3, n),
        n_random_samples=100, return_scores=True
    )
    assert len(scores) == n


def test_gridsearch_random_sampling_invalid(mcycle_X_y):
    """
    invalid n_random_samples should raise ValueError.
    """
    X, y = mcycle_X_y

    with pytest.raises(ValueError):
        LinearGAM().gridsearch(X, y, n_random_samples=0)

    with pytest.raises(ValueError):
        LinearGAM().gridsearch(X, y, n_random_samples=-1)
