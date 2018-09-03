# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pygam.terms import Term, SplineTerm, LinearTerm, FactorTerm, TensorTerm, TermList

def test_wrong_length():
    """iterable params must all match lengths
    """
    with pytest.raises(ValueError):
        SplineTerm(0, lam=[0, 1, 2], penalties=['auto', 'auto'])
