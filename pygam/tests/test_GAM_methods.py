# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pygam import *


def test_LinearGAM_pdeps_shape(wage):
    """
    check that we get the expected number of partial dependence functions
    """
    X, y = wage
    gam = LinearGAM().fit(X, y)
    pdeps = gam.partial_dependence(X)
    assert(X.shape == pdeps.shape)

def test_LinearGAM_prediction(mcycle):
    """
    check that we the predictions we get are correct shape
    """
    X, y = mcycle
    preds = LinearGAM().fit(X, y).predict(X)
    assert(preds.shape == y.shape)

def test_LogisticGAM_accuracy(default):
    """
    check that we can compute accuracy correctly
    """
    X, y = default
    gam = LogisticGAM().fit(X, y)

    preds = gam.predict(X)
    acc0 = (preds == y).mean()
    acc1 = gam.accuracy(X, y)
    assert(acc0 == acc1)

def test_summary(mcycle):
    """
    check that we can get a summary if we've fitted the model, else not
    """
    X, y = mcycle
    gam = LinearGAM()

    try:
      gam.summary()
      assert(False)
    except AttributeError:
      assert(True)

    gam.fit(X, y).summary()
    assert(True)


# TODO test deviance_residuals
# TODO test gam conf intervals
# TODO test linear gam pred intervals
# TODO set params
# TODO get params
