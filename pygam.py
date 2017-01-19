# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from numpy import random as rng
import scipy as sp
from sklearn.base import BaseEstimator
from progressbar import ProgressBar

from copy import deepcopy

def gen_knots(data, n_knots=10, add_boundaries=False):
        """
        generate knots from data quantiles
        """
        knots = np.percentile(data, np.linspace(0,100, n_knots+2))
        if add_boundaries:
            return knots
        return knots[1:-1]

def b_spline_basis(x, boundary_knots, order=4, sparse=True):
    """
    generate b-spline basis using De Boor recursion
    """
    x = np.atleast_2d(x).T
    aug_knots = np.r_[boundary_knots.min() * np.ones(order-1), np.sort(boundary_knots), boundary_knots.max() * np.ones(order-1)]

    bases = (x >= aug_knots[:-1]).astype(np.int) * (x < aug_knots[1:]).astype(np.int) # haar bases
    bases[(x >= aug_knots[-1])[:,0], -order] = 1 # want the last basis function extend past the boundary
    bases[(x < aug_knots[0])[:,0], order] = 1

    maxi = len(aug_knots) - 1

    # do recursion from Hastie et al.
    for m in range(2, order + 1):
        maxi -= 1
        maskleft = aug_knots[m-1:maxi+m-1] == aug_knots[:maxi] # bookkeeping to avoid div by 0
        maskright = aug_knots[m:maxi+m] == aug_knots[1:maxi+1]

        left = ((x - aug_knots[:maxi]) / (aug_knots[m-1:maxi+m-1] - aug_knots[:maxi])) * bases[:,:maxi]
        left[:,maskleft] = 0.

        right = ((aug_knots[m:maxi+m]-x) / (aug_knots[m:maxi+m] - aug_knots[1:maxi+1])) * bases[:,1:maxi+1]
        right[:,maskright] = 0.

        bases = left + right

    if sparse:
        return sp.sparse.csc_matrix(bases)

    return bases

class LogisticGAM(BaseEstimator):
    """
    Logistic Generalized Additive Model

    # TODO
    add standard errors
    add support for custom penalties
    add support for different penalty per feature
    add CV
    add search for best Lambda vector
    add support for categorical features => piecewise constant splines, no difference penaly
    """
    def __init__(self, lam=0.6, n_iter=100, tol=1e-5, n_knots=10, diff_order=1):
        self.n_iter = n_iter
        self.lam = lam
        self.tol = tol
        self.n_knots = n_knots
        self.diff_order = diff_order

        # created by other methods
        self.b_ = None
        self.knots_ = None
        self.n_bases_ = []
        self.edof = None
        self.acc = [] # accuracy log
        self.nll = [] # negative log-likelihood log
        self.diffs = [] # differences log


    def __repr__(self):
        name = self.__class__.__name__
        param_kvs = [(k,v) for k,v in self.get_params().iteritems()]
        params = ', '.join(['{}={}'.format(k, repr(v)) for k,v in param_kvs])
        return "%s(%s)" % (name, params)

    def get_params(self, deep=True):
        exclude = ['edof', 'acc', 'nll', 'diffs']
        return dict([(k,v) for k,v in self.__dict__.iteritems() if k[-1]!='_' and (k not in exclude)])

    def set_params(self, **parameters):
        param_names = self.get_params().keys()
        for parameter, value in parameters.items():
            if parameter in param_names:
                setattr(self, parameter, value)

    @property
    def lambdas(self):
        # penalties
        return np.ones(len(self.n_bases_)) * self.lam # vector of lambdas

    def predict_proba(self, X):
        return self.proba_(self.log_odds_(X))

    def proba_(self, log_odds):
        return 1./(1. + np.exp(-log_odds))

    def log_odds_(self, X, bases=None, b_=None):
        if bases is None:
            bases = self.bases_(X)
        if b_ is None:
            b_ = self.b_
        return bases.dot(b_).squeeze()

    def accuracy_(self, y, proba):
        return ((proba > 0.5).astype(int) == y).mean()

    def bases_(self, X):
        bases = [np.ones((X.shape[0],1))] # intercept
        self.n_bases_ = [1] # keep track of how many basis functions in each spline
        for x, knots in zip(X.T, self.knots_):
            bases.append(b_spline_basis(x, knots, sparse=True))
#             bases[-1] = bases[-1][::-1].T[::-1].T # reverse the bases to see if the problem is indeed in the bases
#             bases[-1][:,:7] = bases[-1][:,7:][::-1].T[::-1].T# make basis symmetric
            self.n_bases_.append(bases[-1].shape[1])
        return sp.sparse.hstack(bases, format='csc')

    def proto_P_(self, n):
        """
        builds a proto-penalty matrix for P-Splines.
        penalizes the squared differences between adjacent basis coefficients.

        TODO make sparse
        """

        if n==1:
            return sp.sparse.csc_matrix(1)
        D = np.diff(np.eye(n), n=self.diff_order)
        return sp.sparse.csc_matrix(D.dot(D.T))

    def P_(self):
        """
        penatly matrix for P-Splines

        builds the GLM block-diagonal penalty matrix out of
        proto-penalty matrices from each feature.

        each proto-penalty matrix is multiplied by a lambda for that feature.
        the first feature is the intercept.

        so for m features:
        P = block_diag[lam0 * P0, lam1 * P1, lam2 * P2, ... , lamm * Pm]
        """
        Ps = [self.proto_P_(n) for n in self.n_bases_]
        P_matrix = sp.sparse.block_diag(tuple([P.multiply(lam) for lam,P in zip(self.lambdas, Ps)]))

        return P_matrix + sp.sparse.diags(np.ones(len(self.b_)) * 1e-7) # improve condition

    def pseudo_data_(self, y, log_odds, proba):
        return log_odds + (y - proba)/(proba*(1-proba))

    def weights_(self, proba):
        return sp.sparse.diags(proba*(1-proba), format='csc')

    def pirls_(self, X, y):
        bases = self.bases_(X) # build a basis matrix for the GLM

        # initialize GLM coefficients
        if self.b_ is None:
            self.b_ = np.zeros((bases.shape[1],1)) # allow more training

        P = self.P_() # create penalty matrix

        for _ in range(self.n_iter):
            log_odds = self.log_odds_(X, bases=bases)
            proba = self.proba_(log_odds)
            self.acc.append(self.accuracy_(y, proba)) # log the training accuracy
            self.nll.append(-self.loglikelohood_(y=y, proba=proba))

            # classic problem with logistic regression
            if (proba == 0.).any() or (proba == 1.).any():
                print 'increase regularization'
                break

            weights = self.weights_(proba) # PIRLS
            pseudo_data = self.pseudo_data_(y, log_odds, proba) # PIRLS

            BW = bases.T.dot(weights).tocsc() # common matrix product
            inner = sp.sparse.linalg.inv(BW.dot(bases) + P) # keep for edof

            b_new = inner.dot(BW).dot(pseudo_data)
            diff = np.linalg.norm(self.b_ - b_new)/np.linalg.norm(b_new)
            self.diffs.append(diff)
            self.b_ = b_new # update

            # check convergence
            if diff < self.tol:
                self.edof = self.estimate_edof_(bases, inner, BW)
                return

        print 'did not converge'

    def estimate_edof_(self, bases, inner, BW):
        """
        approximate effective degrees of freedom

        need to find out a good way of doing this
        for now, let's subsample the data matrices, then scale the trace
        """
        size = bases.shape[0]
        max_ = np.min([5000, size])
        scale = np.float(size)/max_
        idxs = range(size)
        np.random.shuffle(idxs)
        return scale * bases.dot(inner).tocsr()[idxs[:max_]].dot(BW[:,idxs[:max_]]).diagonal().sum()

    def fit(self, X, y):
        self.knots_ = [gen_knots(feat, add_boundaries=True, n_knots=self.n_knots) for feat in X.T]
        self.pirls_(X, y)
        return self

    def predict(self, X):
        return self.predict_proba(X) > 0.5

    def partial_dependence(self, X):
        """
        also want an option to return confidence interval
        """
        bases = []
        p_deps = []
        bs = []

        total = 1
        for x, knots in zip(X.T, self.knots_):
            bases.append(b_spline_basis(x, knots, sparse=True))
#             bases.append(b_spline_basis(x, knots, sparse=True)[::-1].T[::-1].T) #reverse bases
#             b = b_spline_basis(x, knots, sparse=True) # make symm
#             b[:,:7] = b[:,7:][::-1].T[::-1].T # make symm
#             bases.append(b) # make symm
            bs.append(self.b_[total:total+bases[-1].shape[1]])
            total += len(bs[-1])

            p_deps.append(self.log_odds_(x, bases[-1], bs[-1]))

        return np.vstack(p_deps).T

    def likelihood_(self, X, y, proba=None):
        if proba is None:
            proba = self.predict_proba(X)
        return np.sum((proba**y) * (1-proba)**(1-y))

    def loglikelohood_(self, X=None, y=None, proba=None):
        return np.log(self.likelihood_(X, y, proba=proba))

    def aic(self):
        return -2*np.exp(-self.nll) + 2*self.edof
