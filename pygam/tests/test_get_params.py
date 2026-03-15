import numpy as np

from pygam import LinearGAM, f, s


def test_get_params_returns_independent_copies():
    # Issue #522: get_params() exposes mutable nested objects
    # Creating a sample model with a combination of terms
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = np.random.randn(100)

    # Adding string and number parameters
    gam = LinearGAM(s(0) + f(1)).fit(X, y)

    # Retrieve the parameters dict
    params = gam.get_params()

    # Grab the original value for lambda of the first term
    original_lam = gam.terms[0].lam

    # Attempt to mutate the value in the returned dict
    # This should NOT affect the underlying standard model configuration because it should be a deepcopy
    params["terms"][0].lam = 9999.0

    # Assert that the model's actual parameter hasn't changed
    assert gam.terms[0].lam == original_lam, (
        "Modifying get_params() result inappropriately altered the model's term instances"
    )

    # Additionally test array mutation
    if hasattr(gam.terms[0], "edge_knots_"):
        params["terms"][0].edge_knots_ = [100.0, 200.0]
        assert np.any(gam.terms[0].edge_knots_ != [100.0, 200.0]), (
            "Array modifications in get_params exposed model's internal data"
        )


def test_get_params_scalars_not_copied():
    """Immutable scalar params should be returned as-is (no deepcopy overhead)."""
    gam = LinearGAM()
    params = gam.get_params()

    # int, float, bool, str should be identical objects (not deepcopied)
    assert params["max_iter"] is gam.max_iter
    assert params["tol"] is gam.tol
    assert params["verbose"] is gam.verbose
    assert params["fit_intercept"] is gam.fit_intercept


def test_get_params_lists_are_copied():
    """Mutable list params like callbacks should be independent copies."""
    gam = LinearGAM()
    params = gam.get_params()

    assert params["callbacks"] is not gam.callbacks
    assert params["callbacks"] == gam.callbacks
