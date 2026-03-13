import numpy as np

from pygam import LinearGAM, PoissonGAM


def test_outer_lambda_improves_gcv(mcycle_X_y):
    X, y = mcycle_X_y

    # start from a deliberately bad smoothing level
    base = LinearGAM(lam=100.0, max_iter=50, tol=1e-3).fit(X, y)
    base_score = base.statistics_["GCV"]

    tuned = LinearGAM(lam=100.0, max_iter=50, tol=1e-3).fit(
        X,
        y,
        lam_search="outer",
        lam_search_max_iter=8,
        lam_search_tol=1e-4,
        lam_search_objective="GCV",
    )

    assert tuned.statistics_["lam_opt_result"] is not None
    assert tuned.statistics_["lam_opt_result"].success
    assert tuned.statistics_["GCV"] <= base_score * 1.01


def test_outer_lambda_known_scale_uses_ubre(faithful_X_y):
    X, y = faithful_X_y

    gam = PoissonGAM(lam=5.0, max_iter=50, tol=1e-3).fit(
        X,
        y,
        lam_search="outer",
        lam_search_max_iter=6,
        lam_search_objective="UBRE",
    )

    res = gam.statistics_.get("lam_opt_result")
    assert res is not None
    assert res.success or res.nit > 0
    assert gam.statistics_["UBRE"] is not None
    assert np.all(np.asarray(gam.statistics_["lam"]) > 0)
