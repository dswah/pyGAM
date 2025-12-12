import numpy as np

from pygam import LinearGAM, s, te
from pygam.penalties import (
    concave,
    convex,
    derivative,
    l2,
    monotonic_dec,
    monotonic_inc,
    none,
    wrap_penalty,
)


def test_single_spline_penalty():
    """
    check that feature functions with only 1 basis are penalized correctly

    derivative penalty should be 0.
    l2 should penalty be 1.
    monotonic_ and convexity_ should be 0.
    """
    coef = np.array(1.0)
    assert np.all(derivative(1, coef).toarray() == 0.0)
    assert np.all(l2(1, coef).toarray() == 1.0)
    assert np.all(monotonic_inc(1, coef).toarray() == 0.0)
    assert np.all(monotonic_dec(1, coef).toarray() == 0.0)
    assert np.all(convex(1, coef).toarray() == 0.0)
    assert np.all(concave(1, coef).toarray() == 0.0)
    assert np.all(none(1, coef).toarray() == 0.0)


def test_wrap_penalty():
    """
    check that wrap penalty indeed reduces inserts the desired penalty into the
    linear term when fit_linear is True, and 0, when fit_linear is False.
    """
    coef = np.array(1.0)
    n = 2
    linear_penalty = -1

    fit_linear = True
    p = wrap_penalty(none, fit_linear, linear_penalty=linear_penalty)
    P = p(n, coef).toarray()
    assert P.sum() == linear_penalty

    fit_linear = False
    p = wrap_penalty(none, fit_linear, linear_penalty=linear_penalty)
    P = p(n, coef).toarray()
    assert P.sum() == 0.0


def test_monotonic_inc(hepatitis_X_y):
    """
    check that monotonic_inc constraint produces monotonic increasing function
    """
    X, y = hepatitis_X_y

    gam = LinearGAM(terms=s(0, constraints="monotonic_inc"))
    gam.fit(X, y)

    XX = gam.generate_X_grid(term=0)
    Y = gam.predict(np.sort(XX))
    diffs = np.diff(Y, n=1)
    assert ((diffs >= 0) + np.isclose(diffs, 0.0)).all()


def test_monotonic_dec(hepatitis_X_y):
    """
    check that monotonic_dec constraint produces monotonic decreasing function
    """
    X, y = hepatitis_X_y

    gam = LinearGAM(terms=s(0, constraints="monotonic_dec"))
    gam.fit(X, y)

    XX = gam.generate_X_grid(term=0)
    Y = gam.predict(np.sort(XX))
    diffs = np.diff(Y, n=1)
    assert ((diffs <= 0) + np.isclose(diffs, 0.0)).all()


def test_convex(hepatitis_X_y):
    """
    check that convex constraint produces convex function
    """
    X, y = hepatitis_X_y

    gam = LinearGAM(terms=s(0, constraints="convex"))
    gam.fit(X, y)

    XX = gam.generate_X_grid(term=0)
    Y = gam.predict(np.sort(XX))
    diffs = np.diff(Y, n=2)
    assert ((diffs >= 0) + np.isclose(diffs, 0.0)).all()


def test_concave(hepatitis_X_y):
    """
    check that concave constraint produces concave function
    """
    X, y = hepatitis_X_y

    gam = LinearGAM(terms=s(0, constraints="concave"))
    gam.fit(X, y)

    XX = gam.generate_X_grid(term=0)
    Y = gam.predict(np.sort(XX))
    diffs = np.diff(Y, n=2)
    assert ((diffs <= 0) + np.isclose(diffs, 0.0)).all()


def test_OOM_large_penalty_matrices_regression(wage_X_y):
    """
    Regression Test to avoid dense penalty matrices in tensor splines
    """
    X, y = wage_X_y
    try:
        # https://github.com/dswah/pyGAM/issues/294 failed with 100k features, try with 1M
        # tensor splines tried to build a dense penalty matrix, then cast to sparse
        gam = LinearGAM(te(0, 1, n_splines=[1000, 1000]))
        gam.terms[0].build_penalties()
    except MemoryError as e:
        pytest.fail(f"Out of Memory: {str(e)}")


def test_OOM_large_constraint_matrices_regression(wage_X_y):
    """
    Regression Test to avoid dense penalty matrices in tensor splines
    """
    X, y = wage_X_y
    try:
        # tensor splines tried to build a dense penalty matrix, then cast to sparse
        gam = LinearGAM(
            terms=te(
                0,
                1,
                n_splines=[1000, 1000],
                penalties=None,
                constraints=["monotonic_dec", "monotonic_dec"],
            )
        )
    except MemoryError:
        # ingore any possible OOM errors in penalties
        pass

    # explicitly build constraints avoiding dense matrices
    try:
        gam.terms[0].build_constraints(
            coef=np.arange(gam.terms[0].n_coefs),
            constraint_lam=1.0,
            constraint_l2=0.00001,
        )
    except MemoryError as e:
        pytest.fail(f"Out of Memory: {str(e)}")


def test_OOM_none_penalty():
    # avoid all dense computation in None penalty
    none(1_000_000, coef=None)


# TODO penalties gives expected matrix structure
# TODO circular constraints
