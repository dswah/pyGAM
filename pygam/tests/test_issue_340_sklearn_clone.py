import numpy as np
import pytest

# Skip this entire test module if scikit-learn is not installed
pytest.importorskip("sklearn")

from sklearn.base import clone

from pygam import LinearGAM, LogisticGAM
from pygam.terms import l, s


def test_sklearn_clone_preserves_terms():
    """
    Test that sklearn.base.clone() properly reconstructs a pyGAM estimator
    and preserves its terms, even after it has been fully fitted.
    See: https://github.com/dswah/pyGAM/issues/340
    """
    X = np.random.rand(50, 3)
    y = np.random.rand(50)

    # Test 1: LinearGAM with default string terms ('auto')
    gam1 = LinearGAM()
    gam1.fit(X, y)
    gam1_cloned = clone(gam1)

    # After cloning, the new instance should have the original string 'auto' terms,
    # NOT a resolved TermList, and should NOT be fitted.
    assert hasattr(gam1_cloned, "terms")
    assert gam1_cloned.terms == "auto"
    assert not gam1_cloned._is_fitted

    # It should be fitable and match original predictions closely
    gam1_cloned.fit(X, y)
    assert np.allclose(gam1.predict(X), gam1_cloned.predict(X))

    # Test 2: LinearGAM with custom TermList
    custom_terms = s(0) + l(1) + s(2, n_splines=20)
    gam2 = LinearGAM(custom_terms)
    gam2.fit(X, y)
    gam2_cloned = clone(gam2)

    # After cloning, terms should be the original TermList passed in
    assert repr(gam2_cloned.terms) == repr(custom_terms)

    # Test 3: LogisticGAM with binary target
    y_bin = (y > 0.5).astype(float)
    gam3 = LogisticGAM()
    gam3.fit(X, y_bin)
    gam3_cloned = clone(gam3)

    assert gam3_cloned.terms == "auto"
    gam3_cloned.fit(X, y_bin)
    assert np.allclose(gam3.predict(X), gam3_cloned.predict(X))
