import numpy as np

import pytest

from pygam import LinearGAM, s, f, te, intercept

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
    assert gam.terms[0].lam == original_lam, "Modifying get_params() result inappropriately altered the model's term instances"
    
    # Additionally test array mutation
    if hasattr(gam.terms[0], "edge_knots_"):
        params["terms"][0].edge_knots_ = [100.0, 200.0]
        assert np.any(gam.terms[0].edge_knots_ != [100.0, 200.0]), "Array modifications in get_params exposed model's internal data"

