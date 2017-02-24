# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from numpy import random as rng
import scipy as sp
from sklearn.base import BaseEstimator
from progressbar import ProgressBar

from copy import deepcopy


EPS = np.finfo(np.float64).eps # machine epsilon

def check_dtype_(X):
    jitter = np.random.randn(X.shape[0])
    dtypes_ = []
    for feat in X.T:
        dtype = feat.dtype.type
        assert issubclass(dtype, (np.int, np.float)), 'data must be discrete or continuous valued'

        if issubclass(dtype, np.int) or (len(np.unique(feat)) != len(np.unique(feat + jitter))):
            assert (np.max(feat) - np.min(feat)) == (len(np.unique(feat)) - 1), 'k categories must be mapped to integers in [0, k-1] interval'
            dtypes_.append(np.int)
            continue

        if issubclass(dtype, np.float):
            dtypes_.append(np.float)
            continue
    return dtypes_

def gen_knots(data, dtype, n_knots=10, add_boundaries=False):
        """
        generate knots from data quantiles

        for discrete data, assumes k categories in [0, k-1] interval
        """
        assert dtype in [np.int, np.float], 'unsupported dtype'
        if dtype == np.int:
            knots = np.r_[np.min(data) - 0.5, np.unique(data) + 0.5]
        else:
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


class LogisticGAM(object):
    """
    Logistic Generalized Additive Model
    """
    def __init__(self, lam=0.6, n_iter=100, n_knots=20, spline_order=4,
                 penalty_matrix='auto', tol=1e-5):

        assert (n_iter >= 1) and (type(n_iter) is int), 'n_iter must be int >= 1'

        self.n_iter = n_iter
        self.tol = tol
        self.lam = lam
        self.n_knots = n_knots
        self.spline_order = spline_order
        self.penalty_matrix = penalty_matrix

        # created by other methods
        self.b_ = None
        self.n_bases_ = []
        self.knots_ = []
        self.lam_ = []
        self.n_knots_ = []
        self.spline_order_ = []
        self.penalty_matrix_ = []
        self.dtypes_ = []
        self.opt_ = 0 # use 0 for numerically stable optimizer, 1 for naive

        # statistics and logging
        self.edof_ = None # effective degrees of freedom
        self.se_ = None # standard errors
        self.aic_ = None # AIC
        self.aicc_ = None # corrected AIC
        self.acc = [] # accuracy log
        self.nll = [] # negative log-likelihood log
        self.diffs = [] # differences log

    def __repr__(self):
        name = self.__class__.__name__
        param_kvs = [(k,v) for k,v in self.get_params().iteritems()]
        params = ', '.join(['{}={}'.format(k, repr(v)) for k,v in param_kvs])
        return "%s(%s)" % (name, params)

    def get_params(self, deep=True):
        exclude = ['acc', 'nll', 'diffs']
        return dict([(k,v) for k,v in self.__dict__.iteritems() if k[-1]!='_' and (k not in exclude)])

    def set_params(self, **parameters):
        param_names = self.get_params().keys()
        for parameter, value in parameters.items():
            if parameter in param_names:
                setattr(self, parameter, value)
        return self

    def expand_attr_(self, attr, n, dt_alt=None, msg=None):
        """
        if self.attr is a list of values of length n,
        then use it as the expanded version,
        otherwise extend the single value to a list of length n

        dt_alt is an alternative value for dtypes of type integer
        """
        data = getattr(self, attr)

        attr_ = attr + '_'
        if isinstance(data, list):
            assert len(data) == n, msg
            setattr(self, attr_, data)
        else:
            data_ = [data] * n
            if dt_alt is not None:
                data_ = [d if dt != np.int else dt_alt for d,dt in zip(data_, self.dtypes_)]
            setattr(self, attr_, data_)

    def gen_knots_(self, X):
        self.expand_attr_('n_knots', X.shape[1], dt_alt=0, msg='n_knots must have the same length as X.shape[1]')
        assert all([(n_knots >= 0) and (type(n_knots) is int) for n_knots in self.n_knots_]), 'n_knots must be int >= 0'
        self.knots_ = [gen_knots(feat, dtype, add_boundaries=True, n_knots=n) for feat, n, dtype in zip(X.T, self.n_knots_, self.dtypes_)]
        self.n_knots_ = [len(knots) - 2 for knots in self.knots_] # update our number of knots, exclude boundaries

    def log_odds_(self, X=None, bases=None, b_=None):
        if bases is None:
            bases = self.bases_(X)
        if b_ is None:
            b_ = self.b_
        return bases.dot(b_).flatten()

    def proba_(self, log_odds):
        return 1./(1. + np.exp(-log_odds))

    def predict_proba(self, X):
        return self.proba_(self.log_odds_(X))

    def accuracy(self, X=None, y=None, proba=None):
        if proba is None:
            proba = self.predict_proba(X)
        return ((proba > 0.5).astype(int) == y).mean()

    def predict(self, X):
        return self.predict_proba(X) > 0.5

    def bases_(self, X):
        """
        Build a matrix of spline bases for each feature, and stack them horizontally

        B = [B_0, B_1, ..., B_p]
        """
        bases = [np.ones((X.shape[0], 1))] # intercept
        self.n_bases_ = [1] # keep track of how many basis functions in each spline
        for x, knots, order in zip(X.T, self.knots_, self.spline_order_):
            bases.append(b_spline_basis(x, knots, sparse=True, order=order))
            self.n_bases_.append(bases[-1].shape[1])
        return sp.sparse.hstack(bases, format='csc')

    def cont_P_(self, n, diff_order=1):
        """
        builds a default proto-penalty matrix for P-Splines for continuous features.
        penalizes the squared differences between adjacent basis coefficients.
        """
        if n==1:
            return sp.sparse.csc_matrix(0.) # no second order derivative for constant functions
        D = np.diff(np.eye(n), n=diff_order)
        return sp.sparse.csc_matrix(D.dot(D.T))

    def cat_P_(self, n):
        """
        builds a default proto-penalty matrix for P-Splines for categorical features.
        penalizes the squared value of each basis coefficient.
        """
        return sp.sparse.csc_matrix(np.eye(n))

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
        Ps = [pmat(n) if pmat not in ['auto', None] else self.cont_P_(n) for n, pmat in zip(self.n_bases_, self.penalty_matrix_)]
        P_matrix = sp.sparse.block_diag(tuple([np.multiply(P, lam) for lam, P in zip(self.lam_, Ps)]))

        return P_matrix

    def pseudo_data_(self, y, log_odds, proba):
        return log_odds + (y - proba)/(proba*(1-proba))

    def weights_(self, proba):
        sqrt_proba = proba ** 0.5
        return sp.sparse.diags(sqrt_proba*(1-sqrt_proba), format='csc')

    def mask_(self, proba):
        mask = (proba != 0) * (proba != 1)
        assert mask.sum() != 0, 'increase regularization'
        return mask

    def pirls_(self, X, y):
        bases = self.bases_(X) # build a basis matrix for the GLM
        m = bases.shape[1]

        # initialize GLM coefficients
        if self.b_ is None:
            self.b_ = np.zeros(bases.shape[1]) # allow more training

        P = self.P_() # create penalty matrix
        S = P # + self.H # add any use-chosen penalty to the diagonal
        S += sp.sparse.diags(np.ones(m) * np.sqrt(EPS)) # improve condition

        E = np.linalg.cholesky(S.todense())
        Dinv = np.zeros((2*m, m)).T

        for _ in range(self.n_iter):
            log_odds = self.log_odds_(bases=bases)
            proba = self.proba_(log_odds)

            mask = self.mask_(proba)
            proba = proba[mask] # update
            log_odds = log_odds[mask] # update

            self.acc.append(self.accuracy(y=y[mask], proba=proba)) # log the training accuracy
            self.nll.append(-self.loglikelihood_(y=y[mask], proba=proba)) # log the training deviance

            weights = self.weights_(proba) # PIRLS
            pseudo_data = weights.dot(self.pseudo_data_(y[mask], log_odds, proba)) # PIRLS

            WB = weights.dot(bases[mask,:]) # common matrix product
            Q, R = np.linalg.qr(WB.todense())
            U, d, Vt = np.linalg.svd(np.vstack([R, E.T]))
            svd_mask = d <= (d.max() * np.sqrt(EPS)) # mask out small singular values

            np.fill_diagonal(Dinv, d**-1) # invert the singular values
            U1 = U[:m,:] # keep only top portion of U

            B = Vt.T.dot(Dinv).dot(U1.T).dot(Q.T)
            b_new = B.dot(pseudo_data).A.flatten()
            diff = np.linalg.norm(self.b_ - b_new)/np.linalg.norm(b_new)

            self.b_ = b_new # update
            self.diffs.append(diff) # log the differences

            # check convergence
            if diff < self.tol:
                # self.edof_ = np.dot(U1, U1.T).trace().A.flatten() # this is wrong?
                self.edof_ = self.estimate_edof_(BW=WB.T, inner_BW=B)
                # self.rse_ = self.estimate_rse_(y, proba, weights)
                # self.se_ = self.estimate_se_(bases, inner, WB.T)
                # self.aic_ = self.estimate_AIC_(X, y, proba)
                # self.aicc_ = self.estimate_AICc_(X, y, proba)
                return

        print 'did not converge'

    def pirls_naive_(self, X, y):
        bases = self.bases_(X) # build a basis matrix for the GLM
        m = bases.shape[1]

        # initialize GLM coefficients
        if self.b_ is None:
            self.b_ = np.zeros(bases.shape[1]) # allow more training

        P = self.P_() # create penalty matrix
        P += sp.sparse.diags(np.ones(m) * np.sqrt(EPS)) # improve condition

        for _ in range(self.n_iter):
            log_odds = self.log_odds_(bases=bases)
            proba = self.proba_(log_odds)

            mask = self.mask_(proba)
            proba = proba[mask] # update
            log_odds = log_odds[mask] # update

            self.acc.append(self.accuracy(y=y, proba=proba)) # log the training accuracy
            self.nll.append(-self.loglikelihood_(y=y, proba=proba))

            weights = self.weights_(proba) # PIRLS
            pseudo_data = self.pseudo_data_(y, log_odds, proba) # PIRLS

            BW = bases.T.dot(weights).tocsc() # common matrix product
            inner = sp.sparse.linalg.inv(BW.dot(bases) + P) # keep for edof

            b_new = inner.dot(BW).dot(pseudo_data).flatten()
            diff = np.linalg.norm(self.b_ - b_new)/np.linalg.norm(b_new)
            self.diffs.append(diff)
            self.b_ = b_new # update

            # check convergence
            if diff < self.tol:
                self.edof_ = self.estimate_edof_(bases, inner, BW)
                self.rse_ = self.estimate_rse_(y, proba, weights)
                self.se_ = self.estimate_se_(bases, inner, BW)
                self.aic_ = self.estimate_AIC_(X, y, proba)
                self.aicc_ = self.estimate_AICc_(X, y, proba)
                return

        print 'did not converge'

    def fit(self, X, y):
        # Setup
        n_feats = X.shape[1]

        # set up dtypes
        self.dtypes_ = check_dtype_(X)

        # expand and check lambdas
        self.expand_attr_('lam', n_feats, msg='lam must have the same length as X.shape[1]')
        self.lam_ = [0.] + self.lam_ # add intercept term

        # expand and check spline orders
        self.expand_attr_('spline_order', n_feats, dt_alt=1, msg='spline_order must have the same length as X.shape[1]')
        assert all([(order >= 1) and (type(order) is int) for order in self.spline_order_]), 'spline_order must be int >= 1'

        # expand and check penalty matrices
        self.expand_attr_('penalty_matrix', n_feats, dt_alt=self.cat_P_, msg='penalty_matrix must have the same length as X.shape[1]')
        self.penalty_matrix_ = [p if p != None else 'auto' for p in self.penalty_matrix_]
        self.penalty_matrix_ = ['auto'] + self.penalty_matrix_ # add intercept term
        assert all([(pmat == 'auto') or (callable(pmat)) for pmat in self.penalty_matrix_]), 'penalty_matrix must be callable'

        # set up knots
        self.gen_knots_(X)

        # optimize
        if self.opt_ == 0:
            self.pirls_(X, y)
        if self.opt_ == 1:
            self.pirls_naive_(X, y)
        return self

    def estimate_edof_(self, bases=None, inner=None, BW=None, inner_BW=None):
        """
        estimate effective degrees of freedom

        need to find out a good way of doing this
        for now, let's subsample the data matrices, then scale the trace
        """
        size = BW.shape[1]
        max_ = np.min([5000, size])
        scale = np.float(size)/max_
        idxs = range(size)
        np.random.shuffle(idxs)
        if inner_BW is None:
            return scale * bases.dot(inner).tocsr()[idxs[:max_]].dot(BW[:,idxs[:max_]]).diagonal().sum()
        else:
            return scale * BW[:,idxs[:max_]].T.dot(inner_BW[:,idxs[:max_]]).diagonal().sum()

    def estimate_rse_(self, y, proba, W):
        """
        estimate the residual standard error
        """
        return np.linalg.norm(W.sqrt().dot(y - proba))**2 / (y.shape[0] - self.edof_)

    def estimate_se_(self, bases, inner, BW):
        """
        estimate the standard error of the parameters
        """
        return (inner.dot(BW).dot(bases).dot(inner).diagonal() * self.rse_)**0.5

    def RSE_(self, X, y, W):
        """
        Residual Standard Error

        this is the consistent estimator for the standard deviation of the
        expected value of y|x
        """
        return self.RSS_(X, y, W) / (y.shape[0] - self.edof_)

    def RSS_(self, X, y, W):
        """
        Residual Sum of Squares

        aka sum of squared errors
        """
        return ((y - self.predict(X).dot(W**0.5))**2).sum()

    def loglikelihood_(self, X=None, y=None, proba=None):
        if proba is None:
            proba = self.predict_proba(X)
        return np.sum(y * np.log(proba) + (1-y) * np.log(1-proba))

    def estimate_AIC_(self, X=None, y=None, proba=None):
        """
        Akaike Information Criterion
        """
        return -2*self.loglikelihood_(X, y, proba) + 2*self.edof_

    def estimate_AICc_(self, X=None, y=None, proba=None):
        """
        corrected Akaike Information Criterion
        """
        if self.aic_ is None:
            self.aic_ = self.estimate_AIC_(X, y, proba)
        return self.aic_ + 2*(self.edof_ + 1)*(self.edof_ + 2)/(y.shape[0] - self.edof_ -2)

    def prediction_intervals(self, X):
        pass

    def partial_dependence(self, X):
        """
        also want an option to return confidence interval
        """
        bases = []
        p_deps = []
        bs = []

        total = 1
        for x, knots, order, n_bases in zip(X.T, self.knots_, self.spline_order_, self.n_bases_[1:]):
            bases.append(b_spline_basis(x, knots, sparse=True, order=order))
            bs.append(self.b_[total:total+n_bases])
            p_deps.append(self.log_odds_(x, bases[-1], bs[-1]))
            total += n_bases

        return np.vstack(p_deps).T

    def summary():
        """
        produce a summary of the model statistics including feature significance via F-Test
        """
        pass
