# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np
import pytest

from pygam import *
from pygam.terms import Term, Intercept, SplineTerm, LinearTerm, FactorTerm, TensorTerm, TermList
from pygam.utils import flatten

@pytest.fixture
def chicago_gam(chicago_X_y):
    X, y = chicago_X_y
    gam = PoissonGAM(terms=s(0, n_splines=200) + te(3, 1) + s(2)).fit(X, y)
    return gam

def test_wrong_length():
    """iterable params must all match lengths
    """
    with pytest.raises(ValueError):
        SplineTerm(0, lam=[0, 1, 2], penalties=['auto', 'auto'])

def test_num_coefs(mcycle_X_y, wage_X_y):
    """make sure this method gives correct values
    """
    X, y = mcycle_X_y

    term = Intercept().compile(X)
    assert term.n_coefs == 1

    term = LinearTerm(0).compile(X)
    assert term.n_coefs == 1

    term = SplineTerm(0).compile(X)
    assert term.n_coefs == term.n_splines


    X, y = wage_X_y
    term = FactorTerm(2).compile(X)
    assert term.n_coefs == 5

    term_a = SplineTerm(0).compile(X)
    term_b = SplineTerm(1).compile(X)
    term = TensorTerm(term_a, term_b).compile(X)
    assert term.n_coefs == term_a.n_coefs * term_b.n_coefs

def test_term_list_removes_duplicates():
    """prove that we remove duplicated terms"""
    term = SplineTerm(0)
    term_list = term + term

    assert isinstance(term_list, TermList)
    assert len(term_list) == 1

def test_tensor_invariance_to_scaling(chicago_gam, chicago_X_y):
    """a model with tensor terms should give results regardless of input scaling
    """
    X, y = chicago_X_y
    X[:, 3] = X[:, 3] * 100
    gam = PoissonGAM(terms=s(0, n_splines=200) + te(3, 1) + s(2)).fit(X, y)
    assert np.allclose(gam.coef_, chicago_gam.coef_, atol=1e-6)

def test_tensor_must_have_at_least_2_marginal_terms():
    with pytest.raises(ValueError):
        te(0)

def test_tensor_term_expands_args_to_match_penalties_and_terms():
    tensor = te(0, 1, lam=3)
    assert len(tensor.lam) == 2
    assert len(flatten(tensor.lam)) == 2

    tensor = te(0, 1, penalties='auto')
    assert len(tensor.lam) == 2
    assert len(flatten(tensor.lam)) == 2

    tensor = te(0, 1, penalties=['auto', ['auto', 'auto']])
    assert len(tensor.lam) == 2
    assert len(flatten(tensor.lam)) == 3

def test_tensor_term_skips_kwargs_when_marginal_term_is_supplied():
    tensor = te(0, s(1), n_splines=420)
    assert tensor._terms[0].n_coefs == 420
    assert tensor._terms[1].n_coefs != 420

def test_tensor_term_doesnt_accept_tensor_terms():
    with pytest.raises(ValueError):
        te(l(0), te(0, 1))

def test_tensor_args_length_must_agree_with_number_of_terms():
    with pytest.raises(ValueError):
        te(0, 1, lam=[3])

    with pytest.raises(ValueError):
        te(0, 1, lam=[3])

    with pytest.raises(ValueError):
        te(0, 1, lam=[3, 3, 3])

def test_build_from_info():
    """we can rebuild terms from info
    """
    terms = [Intercept(),
             LinearTerm(0),
             SplineTerm(0),
             FactorTerm(0),
             TensorTerm(0,1)]

    for term in terms:
        assert Term.build_from_info(term.info) == term

    assert te(0, 1) == TensorTerm(SplineTerm(0, n_splines=10), SplineTerm(1, n_splines=10))

def test_by_variable():
    """our fit on the toy tensor dataset with a by variable on the linear feature
    should be similar to the fit with a tensor product of a spline with a linear
    term
    """
    pass

def test_by_variable_doesnt_exist_in_X(mcycle_X_y):
    """raises a value error if we cannot locate the by variable
    """
    term = s(0, by=1)
    with pytest.raises(ValueError):
        term.compile(mcycle_X_y[0])

def test_term_list_from_info():
    """we can remake a term list from info
    """
    term_list = SplineTerm(0) + LinearTerm(1)

    assert Term.build_from_info(term_list.info) == term_list

def test_term_list_only_accepts_terms_or_term_list():
    TermList()
    with pytest.raises(ValueError):
        TermList(None)

def test_pop_term_from_term_list():
    term_list = SplineTerm(0) + LinearTerm(1) + Intercept()
    term_list_2 = deepcopy(term_list)

    # by default we pop the last
    assert term_list_2.pop() == term_list[-1]

    assert term_list_2.pop(0) == term_list[0]

    with pytest.raises(ValueError):
        term_list_2.pop(1) == term_list[0]

def test_no_multiply():
    """trying to multiply terms raises an error
    """
    with pytest.raises(NotImplementedError):
        SplineTerm(0) * LinearTerm(1)

    term_list = SplineTerm(0) + LinearTerm(1)
    with pytest.raises(NotImplementedError):
        term_list * term_list

def test_by_is_similar_to_tensor_with_linear_term(toy_interaction_X_y):
    """for simple interactions we can acheive equivalent fits using:
        - a spline with a by-variable
        - a tensor between spline and a linear term
    """
    X, y = toy_interaction_X_y

    gam_a = LinearGAM(te(s(0, n_splines=20), l(1))).fit(X, y)
    gam_b = LinearGAM(s(0, by=1)).fit(X, y)

    r2_a = gam_a.statistics_['pseudo_r2']['explained_deviance']
    r2_b = gam_b.statistics_['pseudo_r2']['explained_deviance']

    assert np.allclose(r2_a, r2_b)

def test_correct_smoothing_in_tensors(toy_interaction_X_y):
    """check that smoothing penalties are correctly computed across the marginal
    dimensions

    feature 0 is the sinusoid, so this one needs to be wiggly
    feature 1 is the linear function, so this can smoothed heavily
    """
    X, y = toy_interaction_X_y

    # increase smoothing on linear function heavily, to no detriment
    gam = LinearGAM(te(0, 1, lam=[0.6, 10000])).fit(X, y)
    assert gam.statistics_['pseudo_r2']['explained_deviance'] > 0.9

    #  smoothing the sinusoid function heavily reduces fit quality
    gam = LinearGAM(te(0, 1, lam=[10000, 0.6])).fit(X, y)
    assert gam.statistics_['pseudo_r2']['explained_deviance'] < 0.1

class TestRegressions(object):
    def test_no_auto_dtype(self):
        with pytest.raises(ValueError):
            SplineTerm(feature=0, dtype='auto')
