# -*- coding: utf-8 -*-

import sys

import numpy as np
import pytest
import scipy as sp

from pygam import *


class TestPartialDepencence(object):
    def test_partial_dependence_on_univar_data(self, mcycle_X_y):
        """
        partial dependence with univariate data should equal the overall model
        if fit intercept is false
        """
        X, y = mcycle_X_y
        gam = LinearGAM(fit_intercept=False).fit(X, y)
        pred = gam.predict(X)
        pdep = gam.partial_dependence(term=0, X=X)
        assert((pred == pdep.ravel()).all())

    def test_partial_dependence_on_univar_data2(self, mcycle_X_y, mcycle_gam):
        """
        partial dependence with univariate data should NOT equal the overall model
        if fit intercept is false
        """
        X, y = mcycle_X_y
        pred = mcycle_gam.predict(X)
        pdep = mcycle_gam.partial_dependence(term=0, X=X)
        assert((pred != pdep.ravel()).all())

    def test_partial_dependence_feature_doesnt_exist(self, mcycle_gam):
        """
        partial dependence should raise ValueError when requesting a nonexistent
        term
        """
        with pytest.raises(ValueError):
            mcycle_gam.partial_dependence(term=10)

    def test_partial_dependence_gives_correct_shape_no_meshgrid(self, chicago_gam, chicago_X_y):
        """
        when `meshgrid=False`, partial dependence method should return
        - n points if no X is supplied
        - same n_samples as X
        """
        # specify X
        X, y = chicago_X_y
        for i, term in enumerate(chicago_gam.terms):
            if term.isintercept:
                continue

            # no confidence intervals, specify X
            pdep = chicago_gam.partial_dependence(term=i, X=X)
            assert pdep.shape == y.shape

            # no confidence intervals, no X
            pdep = chicago_gam.partial_dependence(term=i)
            assert pdep.shape == (100**len(term),)

            # with confidence intervals, specify X
            pdep, confi = chicago_gam.partial_dependence(term=i, X=X, width=0.95)
            assert pdep.shape == y.shape
            assert confi.shape == (y.shape[0], 2)

            # with confidence intervals, no X
            pdep, confi = chicago_gam.partial_dependence(term=i, width=0.95)
            assert pdep.shape == (100**len(term),)
            assert confi.shape == (100**len(term), 2)

    def test_partial_dependence_gives_correct_shape_with_meshgrid(self, chicago_gam, chicago_X_y):
        """
        when `meshgrid=True`, partial dependence method should return
        - pdep is meshes with the dimension of the term
        - confi is meshes with the dimension of the term, and number of confis
        """
        # specify X
        X, y = chicago_X_y
        for i, term in enumerate(chicago_gam.terms):
            if term.isintercept:
                continue

            # easy way to make a meshgrid to input
            XX = chicago_gam.generate_X_grid(term=i, meshgrid=True, n=50)

            # no confidence intervals, specify X
            pdep = chicago_gam.partial_dependence(term=i, X=XX, meshgrid=True)
            assert pdep.shape == (50,) * len(term)

            # no confidence intervals, no X
            pdep = chicago_gam.partial_dependence(term=i, meshgrid=True)
            assert pdep.shape == (100,) * len(term)

            # with confidence intervals, specify X
            pdep, confi = chicago_gam.partial_dependence(term=i, X=XX, meshgrid=True, width=0.95)
            assert pdep.shape == (50,) * len(term)
            assert confi.shape == (50,) * len(term) + (2,)

            # with confidence intervals, no X
            pdep, confi = chicago_gam.partial_dependence(term=i, meshgrid=True, width=0.95)
            assert pdep.shape == (100,) * len(term)
            assert confi.shape == (100,) * len(term) +(2,)

    def test_partital_dependence_width_and_quantiles_equivalent(self, chicago_gam, chicago_X_y):
        """
        for non-tensor terms, the outputs of `partial_dependence` is the same
        regardless of `meshgrid=True/False`
        """
        assert not chicago_gam.terms[0].istensor
        meshTrue = chicago_gam.partial_dependence(term=0, meshgrid=True)
        meshFalse = chicago_gam.partial_dependence(term=0, meshgrid=False)

        assert (meshTrue == meshFalse).all()

    def test_partial_dependence_meshgrid_true_false_equivalent_for_non_tensors(self, chicago_gam, chicago_X_y):
        """
        for tensor terms the value of `meshgrid` matters
        """
        assert chicago_gam.terms[1].istensor
        meshTrue = chicago_gam.partial_dependence(term=1, meshgrid=True)
        meshFalse = chicago_gam.partial_dependence(term=1, meshgrid=False)

        assert meshTrue.shape != meshFalse.shape
        assert meshTrue.size == meshFalse.size

    def test_intercept_raises_error_for_partial_dependence(self, mcycle_X_y):
        """
        if a user asks for the intercept when none is fitted,
        a ValueError is raised
        """
        X, y = mcycle_X_y

        gam_intercept = LinearGAM(fit_intercept=True).fit(X, y)
        with pytest.raises(ValueError):
            pdeps = gam_intercept.partial_dependence(term=-1)

        gam_no_intercept = LinearGAM(fit_intercept=False).fit(X, y)
        pdeps = gam_no_intercept.partial_dependence(term=-1)

    def test_no_X_needed_for_partial_dependence(self, mcycle_gam):
        """
        partial_dependence() method uses generate_X_grid by default for the X array
        """
        XX = mcycle_gam.generate_X_grid(term=0)
        assert (mcycle_gam.partial_dependence(term=0) == mcycle_gam.partial_dependence(term=0, X=XX)).all()
