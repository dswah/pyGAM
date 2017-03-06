# -*- coding: utf-8 -*-

from __future__ import division
from collections import defaultdict

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

def check_y(y, link, dist):
    y = np.ravel(y)
    assert np.all(~(np.isnan(link.link(y, dist)))), 'y data is not in domain of link function'
    return y

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

def ylogydu(y, u):
    """tool to give desired output for the limit as y -> 0, which is 0"""
    mask = (np.atleast_1d(y)!=0.)
    out = np.zeros_like(u)
    out[mask] = y[mask] * np.log(y[mask] / u[mask])
    return out

class Distribution(object):
    """
    base distribution class
    """
    def __init__(self, name=None, scale=None, levels=None):
        self.name = name
        self.scale = scale

    def __str__(self):
        return "'{}'".format(self.name)

    def __repr__(self):
        name = self.__class__.__name__
        param_kvs = [(k,v) for k,v in self.get_params().iteritems()]
        params = ', '.join(['{}={}'.format(k, repr(v)) for k,v in param_kvs])
        return "%s(%s)" % (name, params)

    def get_params(self):
        exclude = ['name']
        return dict([(k,v) for k,v in self.__dict__.iteritems() if k[-1]!='_' and (k not in exclude)])

    def phi(self, y, mu, edof):
        """
        GLM scale parameter.
        for Binomial and Poisson families this is unity
        for Normal family this is variance
        """
        if self.name in ['binomial','poisson']:
            return 1.
        else:
            return np.sum(self.V(mu**-1) * (y - mu)**2) / (len(mu) - edof)

class NormalDist(Distribution):
    """
    Normal Distribution
    """
    def __init__(self, scale=None, **kwargs):
        super(NormalDist, self).__init__(name='normal', scale=scale, **kwargs)

    def pdf(self, y, mu):
        return np.exp(-(y - mu)**2/(2*self.scale)) / (self.scale * 2 * np.pi)**0.5

    def V(self, mu):
        """glm Variance function"""
        return np.ones_like(mu)

    def deviance(self, y, mu, scaled=True):
        """
        model deviance

        for a gaussian linear model, this is equal to the SSE
        """
        dev = ((y - mu)**2).sum()
        if scaled:
            return dev / self.scale
        return dev

class BinomialDist(Distribution):
    """
    Binomial Distribution
    """
    def __init__(self, levels=1, **kwargs):
        if levels is None:
            levels = 1
        self.levels = levels
        super(BinomialDist, self).__init__(name='binomial', scale=1.)

    def pdf(self, y, mu):
        n = self.levels
        return (sp.misc.comb(n, y) * (mu / n)**y * (1 - (mu / n))**(n - y))

    def V(self, mu):
        """glm Variance function"""
        return mu * (1 - mu/self.levels)

    def deviance(self, y, mu, scaled=True):
        """
        model deviance

        for a bernoulli logistic model, this is equal to the twice the negative loglikelihod.
        """
        dev = 2 * (ylogydu(y, mu) + ylogydu(self.levels - y, self.levels-mu)).sum()
        if scaled:
            return dev / self.scale
        return dev


DISTRIBUTIONS = {'normal': NormalDist,
                 'poisson': None,
                 'binomial': BinomialDist,
                 'gamma': None,
                 'inv_gaussian': None
                 }

class Link(object):
    def __init__(self, name=None):
        self.name = name

    def __str__(self):
        return "'{}'".format(self.name)

    def __repr__(self):
        name = self.__class__.__name__
        param_kvs = [(k,v) for k,v in self.get_params().iteritems()]
        params = ', '.join(['{}={}'.format(k, repr(v)) for k,v in param_kvs])
        return "%s(%s)" % (name, params)

    def get_params(self):
        exclude = ['name']
        return dict([(k,v) for k,v in self.__dict__.iteritems() if k[-1]!='_' and (k not in exclude)])


class IdentityLink(Link):
    def __init__(self):
        super(IdentityLink, self).__init__(name='identity')

    def link(self, mu, dist):
        """
        glm link function
        this is useful for going from mu to the linear prediction
        """
        return mu

    def mu(self, lp, dist):
        """
        glm mean ie inverse of link function
        """
        return lp

    def gradient(self, mu, dist):
        """
        derivative of the linear prediction wrt mu
        """
        return np.ones_like(mu)

class LogitLink(Link):
    def __init__(self):
        super(LogitLink, self).__init__(name='logit')

    def link(self, mu, dist):
        """
        glm link function
        this is useful for going from mu to the linear prediction
        """
        return np.log(mu / (dist.levels - mu))

    def mu(self, lp, dist):
        """
        glm mean ie inverse of link function

        for classification this is the prediction probabilities
        """
        elp = np.exp(lp)
        return dist.levels * elp / (elp + 1)

    def gradient(self, mu, dist):
        """
        derivative of the linear prediction wrt mu
        """
        return dist.levels/(mu*(dist.levels - mu))


LINK_FUNCTIONS = {'identity': IdentityLink,
                  'log': None,
                  'logit': LogitLink,
                  'inverse': None,
                  'inv_squared': None
                  }

# Penalty Matrix Generators
def cont_P(n, diff_order=1):
    """
    builds a default proto-penalty matrix for P-Splines for continuous features.
    penalizes the squared differences between adjacent basis coefficients.
    """
    if n==1:
        return sp.sparse.csc_matrix(0.) # no second order derivative for constant functions
    D = np.diff(np.eye(n), n=diff_order)
    return sp.sparse.csc_matrix(D.dot(D.T))

def cat_P(n):
    """
    builds a default proto-penalty matrix for P-Splines for categorical features.
    penalizes the squared value of each basis coefficient.
    """
    return sp.sparse.csc_matrix(np.eye(n))

# CallBacks
def validate_callback_data(method):
    def method_wrapper(*args, **kwargs):
        expected = method.__code__.co_varnames

        # rename curret gam object
        if 'self' in kwargs:
            gam = kwargs['self']
            del(kwargs['self'])
            kwargs['gam'] = gam

        # loop once to check any missing
        missing = []
        for e in expected:
            if e == 'self':
                continue
            if e not in kwargs:
                missing.append(e)
        assert len(missing) == 0, 'CallBack cannot reference: {}'.format(', '.join(missing))

        # loop again to extract desired
        kwargs_subset = {}
        for e in expected:
            if e == 'self':
                continue
            kwargs_subset[e] = kwargs[e]

        return method(*args, **kwargs_subset)

    return method_wrapper

def validate_callback(callback):
    if not(hasattr(callback, '_validated')) or callback._validated == False:
        assert hasattr(callback, 'on_loop_start') or hasattr(callback, 'on_loop_end'), 'callback must have `on_loop_start` or `on_loop_end` method'
        if hasattr(callback, 'on_loop_start'):
            setattr(callback, 'on_loop_start', validate_callback_data(callback.on_loop_start))
        if hasattr(callback, 'on_loop_end'):
            setattr(callback, 'on_loop_end', validate_callback_data(callback.on_loop_end))
        setattr(callback, '_validated', True)
    return callback

class CallBack(object):
    def __repr__(self):
        return self.__class__.__name__.lower()

@validate_callback
class Deviance(CallBack):
    def on_loop_start(self, gam, y, mu):
        return gam.distribution.deviance(y=y, mu=mu, scaled=False) # log the training deviance

@validate_callback
class Accuracy(CallBack):
    def on_loop_start(self, y, mu):
        return np.mean(y == (mu>0.5))

class Diffs(CallBack):
    def on_loop_end(self, diff):
        return diff

CALLBACKS = {'deviance': Deviance,
             'diffs': Diffs,
             'accuracy': Accuracy
            }


class GAM(object):
    """
    base Generalized Additive Model
    """
    def __init__(self, lam=0.6, n_iter=100, n_knots=20, spline_order=4,
                 penalty_matrix='auto', tol=1e-5, distribution='normal',
                 link='identity', scale=None, levels=None, callbacks=['deviance', 'diffs']):

        assert (n_iter >= 1) and (type(n_iter) is int), 'n_iter must be int >= 1'
        assert [c in ['deviance', 'diffs', 'accuracy'] or issubclass(c, CallBack) for c in callbacks], 'unsupported callback'
        assert distribution in DISTRIBUTIONS, 'distribution not supported'
        assert link in LINK_FUNCTIONS, 'link not supported'

        self.n_iter = n_iter
        self.tol = tol
        self.lam = lam
        self.n_knots = n_knots
        self.spline_order = spline_order
        self.penalty_matrix = penalty_matrix
        self.callbacks = [CALLBACKS[c] if c in CALLBACKS else c for c in callbacks]
        self.callbacks = [validate_callback(c)() for c in self.callbacks]
        self.distribution = DISTRIBUTIONS[distribution](scale=scale, levels=levels)
        self.link = LINK_FUNCTIONS[link]()

        # created by other methods
        self._b = None # model coefficients
        self._n_bases = []
        self._knots = []
        self._lam = []
        self._n_knots = []
        self._spline_order = []
        self._penalty_matrix = []
        self._dtypes = []
        self._opt = 0 # use 0 for numerically stable optimizer, 1 for naive

        # statistics and logging
        self._statistics = None # dict of statistics
        self.logs = defaultdict(list)

        # these are not parameters
        self._exclude = ['logs', 'callbacks']

    def __repr__(self):
        name = self.__class__.__name__
        param_kvs = [(k,v) for k,v in self.get_params().iteritems()]
        params = ', '.join(['{}={}'.format(k, str(v)) for k,v in param_kvs])
        return "%s(%s)" % (name, params)

    def get_params(self, deep=True):
        return dict([(k,v) for k,v in self.__dict__.iteritems() if (k[-1]!='_') and (k[0]!='_') and (k not in self._exclude)])

    def set_params(self, **parameters):
        param_names = self.get_params().keys()
        for parameter, value in parameters.items():
            if parameter in param_names:
                setattr(self, parameter, value)
        return self

    def _expand_attr(self, attr, n, dt_alt=None, msg=None):
        """
        if self.attr is a list of values of length n,
        then use it as the expanded version,
        otherwise extend the single value to a list of length n

        dt_alt is an alternative value for dtypes of type integer (ie discrete)
        """
        data = getattr(self, attr)

        _attr = '_' + attr
        if isinstance(data, list):
            assert len(data) == n, msg
            setattr(self, _attr, data)
        else:
            data_ = [data] * n
            if dt_alt is not None:
                data_ = [d if dt != np.int else dt_alt for d,dt in zip(data_, self._dtypes)]
            setattr(self, _attr, data_)

    def _gen_knots(self, X):
        self._expand_attr('n_knots', X.shape[1], dt_alt=0, msg='n_knots must have the same length as X.shape[1]')
        assert all([(n_knots >= 0) and (type(n_knots) is int) for n_knots in self._n_knots]), 'n_knots must be int >= 0'
        self._knots = [gen_knots(feat, dtype, add_boundaries=True, n_knots=n) for feat, n, dtype in zip(X.T, self._n_knots, self._dtypes)]
        self._n_knots = [len(knots) - 2 for knots in self._knots] # update our number of knots, exclude boundaries

    def _loglikelihood(self, y, mu):
        y = check_y(y, self.link, self.distribution)
        return np.log(self.distribution.pdf(y=y, mu=mu)).sum()

    def _linear_predictor(self, X=None, modelmat=None, b=None, feature=-1):
        """linear predictor"""
        if modelmat is None:
            modelmat = self._modelmat(X, feature=feature)
        if b is None:
            b = self._b[self._select_feature(feature)]
        return modelmat.dot(b).flatten()

    def predict_mu(self, X):
        lp = self._linear_predictor(X)
        return self.link.mu(lp, self.distribution)

    def predict(self, X):
        return self.predict_mu(X)

    def _modelmat(self, X, feature=-1):
        """
        Builds a model matrix, B, out of the spline basis for each feature

        B = [B_0, B_1, ..., B_p]
        """
        assert feature < len(self._n_bases), 'out of range'
        assert feature >=-1, 'out of range'

        if feature == -1:
            modelmat = [np.ones((X.shape[0], 1))] # intercept
            self._n_bases = [1] # keep track of how many basis functions in each spline
            for x, knots, order in zip(X.T, self._knots, self._spline_order):
                modelmat.append(b_spline_basis(x, knots, sparse=True, order=order))
                self._n_bases.append(modelmat[-1].shape[1])
            return sp.sparse.hstack(modelmat, format='csc')

        if feature == 0:
            # intercept
            return sp.sparse.csc_matrix(np.ones((X.shape[0], 1)))

        # return only the basis functions for 1 feature
        return b_spline_basis(X[:,feature-1], self._knots[feature-1], sparse=True, order=self._spline_order[feature-1])

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
        Ps = [pmat(n) if pmat not in ['auto', None] else cont_P(n) for n, pmat in zip(self._n_bases, self._penalty_matrix)]
        P_matrix = sp.sparse.block_diag(tuple([np.multiply(P, lam) for lam, P in zip(self._lam, Ps)]))

        return P_matrix

    def _pseudo_data(self, y, lp, mu):
        return lp + (y - mu) * self.link.gradient(mu, self.distribution)

    def _weights(self, mu):
        """
        TODO lets verify the formula for this.
        if we use the square root of the mu with the stable opt,
        we get the same results as when we use non-sqrt mu with naive opt.

        this makes me think that they are equivalent.

        also, using non-sqrt mu with stable opt gives very small edofs for even lam=0.001
        and the parameter variance is huge. this seems strange to me.

        computed [V * d(link)/d(mu)] ^(-1/2) by hand and the math checks out as hoped.

        ive since moved the square to the naive pirls method to make the code modular.
        """
        return sp.sparse.diags((self.link.gradient(mu, self.distribution)**2 * self.distribution.V(mu=mu))**-0.5)

    def _mask(self, weights):
        mask = (np.abs(weights) >= np.sqrt(EPS)) * (weights != np.nan)
        assert mask.sum() != 0, 'increase regularization'
        return mask

    def _pirls(self, X, Y):
        modelmat = self._modelmat(X) # build a basis matrix for the GLM
        n = modelmat.shape[0]
        m = modelmat.shape[1]

        # initialize GLM coefficients
        if self._b is None:
            self._b = np.zeros(modelmat.shape[1]) # allow more training

        P = self.P_() # create penalty matrix
        S = P # + self.H # add any use-chosen penalty to the diagonal
        S += sp.sparse.diags(np.ones(m) * np.sqrt(EPS)) # improve condition

        E = np.linalg.cholesky(S.todense())
        Dinv = np.zeros((2*m, m)).T

        for _ in range(self.n_iter):
            y = deepcopy(Y) # for simplicity
            lp = self._linear_predictor(modelmat=modelmat)
            mu = self.link.mu(lp, self.distribution)
            weights = self._weights(mu)

            # check for weghts == 0, nan, and update
            mask = self._mask(weights.diagonal())
            y = y[mask] # update
            lp = lp[mask] # update
            mu = mu[mask] # update

            weights = self._weights(mu)
            pseudo_data = weights.dot(self._pseudo_data(y, lp, mu)) # PIRLS Wood pg 183

            # log on-loop-start stats
            self._on_loop_start(vars())

            WB = weights.dot(modelmat[mask,:]) # common matrix product
            Q, R = np.linalg.qr(WB.todense())
            U, d, Vt = np.linalg.svd(np.vstack([R, E.T]))
            svd_mask = d <= (d.max() * np.sqrt(EPS)) # mask out small singular values

            np.fill_diagonal(Dinv, d**-1) # invert the singular values
            U1 = U[:m,:] # keep only top portion of U

            B = Vt.T.dot(Dinv).dot(U1.T).dot(Q.T)
            b_new = B.dot(pseudo_data).A.flatten()
            diff = np.linalg.norm(self._b - b_new)/np.linalg.norm(b_new)
            self._b = b_new # update

            # log on-loop-end stats
            self._on_loop_end(vars())

            # check convergence
            if diff < self.tol:
                # self.edof_ = np.dot(U1, U1.T).trace().A.flatten() # this is wrong?
                self._estimate_model_statistics(Y, modelmat, inner=None, BW=WB.T, B=B)
                return

        print 'did not converge'

    def _pirls_naive(self, X, y):
        modelmat = self._modelmat(X) # build a basis matrix for the GLM
        m = modelmat.shape[1]

        # initialize GLM coefficients
        if self._b is None:
            self._b = np.zeros(modelmat.shape[1]) # allow more training

        P = self.P_() # create penalty matrix
        P += sp.sparse.diags(np.ones(m) * np.sqrt(EPS)) # improve condition

        for _ in range(self.n_iter):
            lp = self._linear_predictor(modelmat=modelmat)
            mu = self.glm_mu_(lp=lp)

            mask = self._mask(mu)
            mu = mu[mask] # update
            lp = lp[mask] # update

            if self.family == 'binomial':
                self.acc.append(self.accuracy(y=y[mask], mu=mu)) # log the training accuracy
            self.dev.append(self.deviance_(y=y[mask], mu=mu, scaled=False)) # log the training deviance

            weights = self._weights(mu)**2 # PIRLS, added square for modularity
            pseudo_data = self._pseudo_data(y, lp, mu) # PIRLS

            BW = modelmat.T.dot(weights).tocsc() # common matrix product
            inner = sp.sparse.linalg.inv(BW.dot(modelmat) + P) # keep for edof

            b_new = inner.dot(BW).dot(pseudo_data).flatten()
            diff = np.linalg.norm(self._b - b_new)/np.linalg.norm(b_new)
            self.diffs.append(diff)
            self._b = b_new # update

            # check convergence
            if diff < self.tol:
                self.edof_ = self._estimate_edof(modelmat, inner, BW)
                self.aic_ = self._estimate_AIC(X, y, mu)
                self.aicc_ = self._estimate_AICc(X, y, mu)
                return

        print 'did not converge'

    def _on_loop_start(self, variables):
        """
        performs on-loop-start actions like callbacks

        variables contains local namespace variables.
        """
        for callback in self.callbacks:
            if hasattr(callback, 'on_loop_start'):
                self.logs[str(callback)].append(callback.on_loop_start(**variables))

    def _on_loop_end(self, variables):
        """
        performs on-loop-end actions like callbacks

        variables contains local namespace variables.
        """
        for callback in self.callbacks:
            if hasattr(callback, 'on_loop_end'):
                self.logs[str(callback)].append(callback.on_loop_end(**variables))

    def fit(self, X, y):
        # Setup
        y = check_y(y, self.link, self.distribution)
        n_feats = X.shape[1]

        # set up dtypes
        self._dtypes = check_dtype_(X)

        # expand and check lambdas
        self._expand_attr('lam', n_feats, msg='lam must have the same length as X.shape[1]')
        self._lam = [0.] + self._lam # add intercept term

        # expand and check spline orders
        self._expand_attr('spline_order', n_feats, dt_alt=1, msg='spline_order must have the same length as X.shape[1]')
        assert all([(order >= 1) and (type(order) is int) for order in self._spline_order]), 'spline_order must be int >= 1'

        # expand and check penalty matrices
        self._expand_attr('penalty_matrix', n_feats, dt_alt=cat_P, msg='penalty_matrix must have the same length as X.shape[1]')
        self._penalty_matrix = [p if p != None else 'auto' for p in self._penalty_matrix]
        self._penalty_matrix = ['auto'] + self._penalty_matrix # add intercept term
        assert all([(pmat == 'auto') or (callable(pmat)) for pmat in self._penalty_matrix]), 'penalty_matrix must be callable'

        # set up knots
        self._gen_knots(X)

        # optimize
        if self._opt == 0:
            self._pirls(X, y)
        if self._opt == 1:
            self._pirls_naive(X, y)
        return self

    def _estimate_model_statistics(self, y, modelmat, inner=None, BW=None, B=None):
        """
        method to compute all of the model statistics
        """
        self._statistics = {}

        lp = self._linear_predictor(modelmat=modelmat)
        mu = self.link.mu(lp, self.distribution)
        self._statistics['edof'] = self._estimate_edof(BW=BW, B=B)
        self.distribution.scale = self.distribution.phi(y=y, mu=mu, edof=self._statistics['edof'])
        self._statistics['cov'] = (B.dot(B.T)).A * self.distribution.scale # parameter covariances. no need to remove a W because we are using W^2. Wood pg 184
        self._statistics['se'] = self._statistics['cov'].diagonal()**0.5
        self._statistics['AIC']= self._estimate_AIC(y=y, mu=mu)
        self._statistics['AICc'] = self._estimate_AICc(y=y, mu=mu)
        self._statistics['pseudo_r2'] = self._estimate_r2(y=y, mu=mu)
        self._statistics['GCV'], self._statistics['UBRE'] = self._estimate_GCV_UBRE(modelmat=modelmat, y=y)

    def _estimate_edof(self, modelmat=None, inner=None, BW=None, B=None, limit=50000):
        """
        estimate effective degrees of freedom.

        computes the only diagonal of the influence matrix and sums.
        allows for subsampling when the number of samples is very large.
        """
        size = BW.shape[1] # number of samples
        max_ = np.min([limit, size]) # since we only compute the diagonal, we can afford larger matrices
        if max_ == limit:
            # subsampling
            scale = np.float(size)/max_
            idxs = range(size)
            np.random.shuffle(idxs)

            if B is None:
                return scale * modelmat.dot(inner).tocsr()[idxs[:max_]].T.multiply(BW[:,idxs[:max_]]).sum()
            else:
                return scale * BW[:,idxs[:max_]].multiply(B[:,idxs[:max_]]).sum()
        else:
            # no subsampling
            if B is None:
                return modelmat.dot(inner).T.multiply(BW).sum()
            else:
                return BW.multiply(B).sum()

    def _estimate_AIC(self, y=None, mu=None):
        """
        Akaike Information Criterion
        """
        estimated_scale = not(self.distribution.name in ['binomial', 'poisson']) # if we estimate the scale, that adds 2 dof
        return -2*self._loglikelihood(y=y, mu=mu) + 2*self._statistics['edof'] + 2*estimated_scale

    def _estimate_AICc(self, X=None, y=None, mu=None):
        """
        corrected Akaike Information Criterion
        """
        edof = self._statistics['edof']
        if self._statistics['AIC'] is None:
            self._statistics['AIC'] = self._estimate_AIC(X, y, mu)
        return self._statistics['AIC'] + 2*(edof + 1)*(edof + 2)/(y.shape[0] - edof -2)

    def _estimate_r2(self, X=None, y=None, mu=None):
        """
        estimate some pseudo R^2 values
        """
        if mu is None:
            mu = self.glm_mu_(X=X)

        n = len(y)
        null_mu = y.mean() * np.ones_like(y)

        null_l = self._loglikelihood(y=y, mu=null_mu)
        full_l = self._loglikelihood(y=y, mu=mu)

        r2 = {}
        r2['mcfadden'] = 1. - full_l/null_l
        r2['mcfadden_adj'] = 1. - (full_l-self._statistics['edof'])/null_l
        r2['cox_snell']= (1. - np.exp(2./n * (null_l - full_l)))
        r2['nagelkerke'] = r2['cox_snell'] / (1. - np.exp(2./n * null_l))
        return r2

    def _estimate_GCV_UBRE(self, X=None, y=None, modelmat=None, gamma=10., add_scale=True):
        """
        Generalized Cross Validation and Un-Biased Risk Estimator.

        UBRE is used when the scale parameter is known, like Poisson and Binomial families.

        Parameters
        ----------
        add_scale:
            boolean. UBRE score can be negative because the distribution scale is subtracted.
            to keep things positive we can add the scale back.
            default: True
        gamma:
            float. serves as a weighting to increase the impact of the influence matrix on the score:
            default: 10.

        Returns
        -------
        score:
            float. Either GCV or UBRE, depending on if the scale parameter is known.

        Notes
        -----
        Sometimes the GCV or UBRE selected model is deemed to be too wiggly,
        and a smoother model is desired. One way to achieve this, in a systematic way, is to
        increase the amount that each model effective degree of freedom counts, in the GCV
        or UBRE score, by a factor γ ≥ 1

        see Wood pg. 177-182 for more details.
        """
        assert gamma >= 1., 'scaling should be greater than 1'

        if modelmat is None:
            modelmat = self._modelmat(X)

        lp = self._linear_predictor(modelmat=modelmat)
        mu = self.link.mu(lp, self.distribution)
        n = y.shape[0]
        edof = self._statistics['edof']

        GCV = None
        UBRE = None

        if self.distribution.name in ['binomial', 'poisson']:
            # scale is known, use UBRE
            scale = self.distribution.scale
            UBRE = 1./n * self.distribution.deviance(mu=mu, y=y) - (~add_scale)*(scale) + 2.*gamma/n * edof * scale
        else:
            # scale unkown, use GCV
            GCV = (n * self.distribution.deviance(mu=mu, y=y)) / (n - gamma * edof)**2
        return (GCV, UBRE)

    def prediction_intervals(self, X, width=.95, quantiles=None):
        return self._get_quantiles(X, width, quantiles, prediction=True)

    def confidence_intervals(self, X, width=.95, quantiles=None):
        return self._get_quantiles(X, width, quantiles, prediction=False)

    def _get_quantiles(self, X, width, quantiles, B=None, lp=None, prediction=False, xform=True, feature=-1):
        if quantiles is not None:
            if issubclass(quantiles.__class__, (np.int, np.float)):
                quantiles = [quantiles]
        else:
            alpha = (1 - width)/2.
            quantiles = [alpha, 1 - alpha]
        for quantile in quantiles:
            assert (quantile**2 <= 1.), 'quantiles must be in [0, 1]'

        if B is None:
            B = self._modelmat(X, feature=feature)
        if lp is None:
            lp = self._linear_predictor(modelmat=B, feature=feature)
        idxs = self._select_feature(feature)
        cov = self._statistics['cov'][idxs][:,idxs]

        scale = self.distribution.scale
        var = (B.dot(cov) * B.todense().A).sum(axis=1) * scale
        if prediction:
            var += scale

        lines = []
        for quantile in quantiles:
            t = sp.stats.t.ppf(quantile, df=self._statistics['edof'])
            lines.append(lp + t * var**0.5)

        if xform:
            return self.link.mu(np.vstack(lines).T, self.distribution)
        return np.vstack(lines).T

    def _select_feature(self, i):
        """
        tool for indexing by feature function.

        many coefficients and parameters are organized by feature.
        this tool returns all of the indices for a given feature.

        GAM intercept is considered the 0th feature.
        """
        assert i < len(self._n_bases), 'out of range'
        assert i >=-1, 'out of range'

        if i == -1:
            # special case for selecting all features
            return np.arange(np.sum(self._n_bases), dtype=int)

        a = np.sum(self._n_bases[:i])
        b = np.sum(self._n_bases[i])
        return np.arange(a, a+b, dtype=int)

    def partial_dependence(self, X, features=None, width=.95, quantiles=None):
        """
        Computes the feature functions for the GAM as well as their confidence intervals.
        """
        m = X.shape[1]
        p_deps = []
        conf_intervals = []

        if features is None:
            features = np.arange(m) + 1 # skips the intercept
        if issubclass(features.__class__, (np.int, np.float)):
            features = np.array([features])

        assert (features >= 0).all() and (features <= m).all(), 'out of range'

        for i in features:
            B = self._modelmat(X, feature=i)
            lp = self._linear_predictor(modelmat=B, feature=i)
            p_deps.append(lp)
            conf_intervals.append(self._get_quantiles(X, width=width,
                                                      quantiles=quantiles,
                                                      B=B, lp=lp,
                                                      feature=i, xform=False))

        return np.vstack(p_deps).T, conf_intervals

    def summary():
        """
        produce a summary of the model statistics including feature significance via F-Test
        """
        pass


class LogisticGAM(GAM):
    """
    Logistic GAM model
    """
    def __init__(self, **kwargs):
        super(LogisticGAM, self).__init__(levels=1,
                                          distribution='binomial',
                                          link='logit',
                                          callbacks=['deviance',
                                                     'diffs',
                                                     'accuracy'],
                                          **kwargs)

        self._exclude += ['distribution', 'link']

    def accuracy(self, X=None, y=None, mu=None):
        if mu is None:
            mu = self.predict_mu(X)
        y = check_y(y, self.link, self.distribution)
        return ((mu > 0.5).astype(int) == y).mean()

    def predict(self, X):
        return self.predict_mu(X) > 0.5

    def predict_proba(self, X):
        return self.predict_mu(X)
