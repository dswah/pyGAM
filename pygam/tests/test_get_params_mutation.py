import numpy as np
from pygam import LinearGAM, s

def test_get_params_does_not_mutate_model():
    X = np.random.rand(50, 1)
    y = np.random.rand(50)

    gam = LinearGAM(s(0)).fit(X, y)

    params = gam.get_params()
    params["terms"][0].lam = 999

    assert gam.terms[0].lam != 999