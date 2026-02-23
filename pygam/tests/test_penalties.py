import numpy as np
import pytest

from pygam import LinearGAM, s, te
from pygam.penalties import (
    concave,
    convex,
    derivative,
    l2,
    monotonic_dec,
    monotonic_inc,
    non_negative,
    non_positive,
    none,
    wrap_penalty,
)


def test_single_spline_penalty():
    """
    check that feature functions with only 1 basis are penalized correctly

    derivative penalty should be 0.
    l2 should penalty be 1.
    monotonic_ and convexity_ should be 0.
    sign constraints should be 0 when coef >= 0 (non_negative) or <= 0 (non_positive).
    """
    coef = np.array(1.0)
    assert np.all(derivative(1, coef).toarray() == 0.0)
    assert np.all(l2(1, coef).toarray() == 1.0)
    assert np.all(monotonic_inc(1, coef).toarray() == 0.0)
    assert np.all(monotonic_dec(1, coef).toarray() == 0.0)
    assert np.all(convex(1, coef).toarray() == 0.0)
    assert np.all(concave(1, coef).toarray() == 0.0)
    assert np.all(none(1, coef).toarray() == 0.0)
    # positive coef satisfies non_negative; no penalty
    assert np.all(non_negative(1, coef).toarray() == 0.0)
    # positive coef violates non_positive; penalty of 1
    assert np.all(non_positive(1, coef).toarray() == 1.0)

    coef_neg = np.array(-1.0)
    # negative coef violates non_negative; penalty of 1
    assert np.all(non_negative(1, coef_neg).toarray() == 1.0)
    # negative coef satisfies non_positive; no penalty
    assert np.all(non_positive(1, coef_neg).toarray() == 0.0)


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


class TestNonNegativePenalty:
    """Unit and integration tests for the non_negative constraint."""

    def test_penalty_matrix_shape(self):
        """Penalty matrix must be square with side n."""
        coef = np.array([1.0, -2.0, 3.0, -4.0])
        P = non_negative(4, coef)
        assert P.shape == (4, 4)

    def test_penalty_matrix_is_diagonal(self):
        """Penalty matrix must be diagonal."""
        coef = np.array([1.0, -2.0, 3.0, -4.0])
        P = non_negative(4, coef).toarray()
        off = P - np.diag(np.diag(P))
        assert np.all(off == 0.0)

    def test_penalty_on_negative_coefs(self):
        """Negative coefficients receive penalty 1; positive receive 0."""
        coef = np.array([1.0, -2.0, 3.0, -4.0])
        diag = non_negative(4, coef).diagonal()
        expected = np.array([0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(diag, expected)

    def test_no_penalty_when_all_positive(self):
        """All-positive coefficients yield a zero penalty matrix."""
        coef = np.array([1.0, 2.0, 3.0])
        P = non_negative(3, coef)
        assert P.nnz == 0

    def test_dimension_mismatch_raises(self):
        """Mismatched n and coef length must raise ValueError."""
        coef = np.array([1.0, -1.0])
        with pytest.raises(ValueError, match="dimension mismatch"):
            non_negative(5, coef)

    def test_fitted_partial_dependence_is_non_negative(self):
        """
        Spline fitted with non_negative constraint should have
        non-negative partial dependence at all grid points.

        B-spline functions are non-negative on their support, so
        non-negative coefficients imply a non-negative feature function.
        """
        rng = np.random.default_rng(0)
        n = 200
        x = np.linspace(0, 2 * np.pi, n)
        # response is always non-negative, so the spline should be too
        y = np.abs(np.sin(x)) + rng.normal(0, 0.1, n)
        X = x.reshape(-1, 1)

        gam = LinearGAM(s(0, n_splines=20, constraints="non_negative")).fit(X, y)
        XX = gam.generate_X_grid(term=0)
        pd = gam.partial_dependence(term=0, X=XX)
        # allow a small numerical tolerance
        assert np.all(pd >= -1e-10), f"min pd = {pd.min():.4g}"

    def test_combination_with_monotonic(self):
        """
        Combining non_negative with monotonic_inc should produce a
        simultaneously monotonic and non-negative feature function.
        """
        rng = np.random.default_rng(1)
        n = 200
        x = np.linspace(0, 3, n)
        y = x**2 + rng.normal(0, 0.2, n)
        X = x.reshape(-1, 1)

        gam = LinearGAM(
            s(0, n_splines=15, constraints=["monotonic_inc", "non_negative"])
        ).fit(X, y)

        XX = gam.generate_X_grid(term=0)
        pd = gam.partial_dependence(term=0, X=XX)
        diffs = np.diff(pd.ravel())
        # monotonically non-decreasing
        assert np.all(diffs >= -1e-10), f"min diff = {diffs.min():.4g}"
        # non-negative
        assert np.all(pd >= -1e-10), f"min pd = {pd.min():.4g}"


class TestNonPositivePenalty:
    """Unit and integration tests for the non_positive constraint."""

    def test_penalty_matrix_shape(self):
        coef = np.array([1.0, -2.0, 3.0, -4.0])
        P = non_positive(4, coef)
        assert P.shape == (4, 4)

    def test_penalty_on_positive_coefs(self):
        """Positive coefficients receive penalty 1; negative receive 0."""
        coef = np.array([1.0, -2.0, 3.0, -4.0])
        diag = non_positive(4, coef).diagonal()
        expected = np.array([1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(diag, expected)

    def test_no_penalty_when_all_negative(self):
        """All-negative coefficients yield a zero penalty matrix."""
        coef = np.array([-1.0, -2.0, -3.0])
        P = non_positive(3, coef)
        assert P.nnz == 0

    def test_dimension_mismatch_raises(self):
        coef = np.array([1.0, -1.0])
        with pytest.raises(ValueError, match="dimension mismatch"):
            non_positive(5, coef)

    def test_fitted_partial_dependence_is_non_positive(self):
        """
        Spline fitted with non_positive constraint should have
        non-positive partial dependence at all grid points.
        """
        rng = np.random.default_rng(2)
        n = 200
        x = np.linspace(0, 2 * np.pi, n)
        y = -np.abs(np.sin(x)) + rng.normal(0, 0.1, n)
        X = x.reshape(-1, 1)

        gam = LinearGAM(s(0, n_splines=20, constraints="non_positive")).fit(X, y)
        XX = gam.generate_X_grid(term=0)
        pd = gam.partial_dependence(term=0, X=XX)
        assert np.all(pd <= 1e-10), f"max pd = {pd.max():.4g}"


class TestSignConstraintInCONSTRAINTS:
    """Verify the new constraints are discoverable via the CONSTRAINTS registry."""

    def test_non_negative_in_constraints_dict(self):
        from pygam.penalties import CONSTRAINTS

        assert "non_negative" in CONSTRAINTS

    def test_non_positive_in_constraints_dict(self):
        from pygam.penalties import CONSTRAINTS

        assert "non_positive" in CONSTRAINTS

    def test_non_negative_by_string_in_spline_term(self):
        """SplineTerm must accept 'non_negative' as a constraint string."""
        rng = np.random.default_rng(3)
        X = rng.random((50, 1))
        y = rng.random(50)
        gam = LinearGAM(s(0, constraints="non_negative")).fit(X, y)
        assert gam._is_fitted

    def test_non_positive_by_string_in_spline_term(self):
        """SplineTerm must accept 'non_positive' as a constraint string."""
        rng = np.random.default_rng(4)
        X = rng.random((50, 1))
        y = -rng.random(50)
        gam = LinearGAM(s(0, constraints="non_positive")).fit(X, y)
        assert gam._is_fitted


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
