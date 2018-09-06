# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pygam import *
from pygam.terms import Term, Intercept, SplineTerm, LinearTerm, FactorTerm, TensorTerm, TermList

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
    assert np.allclose(gam.coef_, chicago_gam.coef_)

def test_tensor_gives_correct_default_n_splines():
    """
    """
    pass

def test_build_from_info():
    terms = [Intercept(),
             LinearTerm(0),
             SplineTerm(0),
             FactorTerm(0),
             TensorTerm(0,1)]

    for term in terms:
        info = term.info
        assert Term.build_from_info(info).info == info

    assert te(0, 1).info == TensorTerm(SplineTerm(0, n_splines=10), SplineTerm(1, n_splines=10)).info
