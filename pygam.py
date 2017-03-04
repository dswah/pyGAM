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

class GAM(object):
    """
    base Generalized Additive Model
    """
    def __init__(self, lam=0.6, n_iter=100, n_knots=20, spline_order=4,
                 penalty_matrix='auto', tol=1e-5, distribution='normal',
                 link='identity', scale=None, levels=None):

        assert (n_iter >= 1) and (type(n_iter) is int), 'n_iter must be int >= 1'

        self.n_iter = n_iter
        self.tol = tol
        self.lam = lam
        self.n_knots = n_knots
        self.spline_order = spline_order
        self.penalty_matrix = penalty_matrix

        assert distribution in DISTRIBUTIONS, 'distribution not supported'
        self.distribution = DISTRIBUTIONS[distribution](scale=scale, levels=levels)
        assert link in LINK_FUNCTIONS, 'link not supported'
        self.link = LINK_FUNCTIONS[link]()

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
        self.cov_ = None # parameter covariance matrix
        self.scale_ = None # estimated scale, eg variance for normal models
        self.gcv_ = None # generalized cross validation
        self.ubre_ = None # unbiased risk estimator
        self.r2_mcfadden_ = None # mcfadden's r^2
        self.r2_mcfadden_adj_ = None # mcfadden's adjusted r^2
        self.r2_cox_snell_ = None # cox & snell r^2
        self.r2_nagelkerke_ = None # nagelkerke's r^2
        self.acc = [] # accuracy log
        self.dev = [] # unscaled deviance log
        self.diffs = [] # differences log

        # these are not parameters
        self._exclude = ['acc', 'dev', 'diffs', 'likelihood']

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

    def loglikelihood_(self, y, mu):
        return np.log(self.distribution.pdf(y=y.ravel(), mu=mu)).sum()

    def linear_predictor_(self, X=None, bases=None, b=None, feature=-1):
        """linear predictor"""
        if bases is None:
            bases = self.bases_(X, feature=feature)
        if b is None:
            b = self.b_[self.select_feature_(feature)]
        return bases.dot(b).flatten()

    def predict_mu(self, X):
        lp = self.linear_predictor_(X)
        return self.link.mu(lp, self.distribution)

    def predict(self, X):
        return self.predict_mu(X)

    def bases_(self, X, feature=-1):
        """
        Build a matrix of spline bases for each feature, and stack them horizontally

        B = [B_0, B_1, ..., B_p]
        """
        assert feature < len(self.n_bases_), 'out of range'
        assert feature >=-1, 'out of range'

        if feature == -1:
            bases = [np.ones((X.shape[0], 1))] # intercept
            self.n_bases_ = [1] # keep track of how many basis functions in each spline
            for x, knots, order in zip(X.T, self.knots_, self.spline_order_):
                bases.append(b_spline_basis(x, knots, sparse=True, order=order))
                self.n_bases_.append(bases[-1].shape[1])
            return sp.sparse.hstack(bases, format='csc')

        if feature == 0:
            # intercept
            return sp.sparse.csc_matrix(np.ones((X.shape[0], 1)))

        # return only the basis functions for 1 feature
        return b_spline_basis(X[:,feature-1], self.knots_[feature-1], sparse=True, order=self.spline_order_[feature-1])

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

    def pseudo_data_(self, y, lp, mu):
        return lp + (y - mu) * self.link.gradient(mu, self.distribution)

    def weights_(self, mu):
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

    def mask_(self, weights):
        mask = (np.abs(weights) >= np.sqrt(EPS)) * (weights != np.nan)
        assert mask.sum() != 0, 'increase regularization'
        return mask

    def pirls_(self, X, y):
        bases = self.bases_(X) # build a basis matrix for the GLM
        n = bases.shape[0]
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
            lp = self.linear_predictor_(bases=bases)
            mu = self.link.mu(lp, self.distribution)
            weights = self.weights_(mu)

            # check for weghts == 0, nan, and update
            mask = self.mask_(weights.diagonal())
            mu = mu[mask] # update
            lp = lp[mask] # update
            weights = self.weights_(mu)
            pseudo_data = weights.dot(self.pseudo_data_(y[mask], lp, mu)) # PIRLS Wood pg 183

            # logs
            # if self.distribution.name == 'binomial':
            #     self.acc.append(self.accuracy(y=y[mask], proba=mu)) # log the training accuracy
            self.dev.append(self.distribution.deviance(y=y[mask], mu=mu, scaled=False)) # log the training deviance

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
                lp = self.linear_predictor_(bases=bases)
                mu = self.link.mu(lp, self.distribution)
                # self.edof_ = np.dot(U1, U1.T).trace().A.flatten() # this is wrong?
                self.edof_ = self.estimate_edof_(BW=WB.T, inner_BW=B)
                self.distribution.scale = self.distribution.phi(y=y, mu=mu, edof=self.edof_)
                self.cov_ = (B.dot(B.T)).A * self.distribution.scale # parameter covariances. no need to remove a W because we are using W^2. Wood pg 184
                self.se_ = self.cov_.diagonal()**0.5
                self.aic_ = self.estimate_AIC_(y=y, mu=mu)
                self.aicc_ = self.estimate_AICc_(y=y, mu=mu)
                self.estimate_GCV_UBRE_(bases=bases, y=y)
                self.estimate_r2_(y=y, mu=mu)
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
            lp = self.linear_predictor_(bases=bases)
            mu = self.glm_mu_(lp=lp)

            mask = self.mask_(mu)
            mu = mu[mask] # update
            lp = lp[mask] # update

            if self.family == 'binomial':
                self.acc.append(self.accuracy(y=y[mask], proba=mu)) # log the training accuracy
            self.dev.append(self.deviance_(y=y[mask], mu=mu, scaled=False)) # log the training deviance

            weights = self.weights_(mu)**2 # PIRLS, added square for modularity
            pseudo_data = self.pseudo_data_(y, lp, mu) # PIRLS

            BW = bases.T.dot(weights).tocsc() # common matrix product
            inner = sp.sparse.linalg.inv(BW.dot(bases) + P) # keep for edof

            b_new = inner.dot(BW).dot(pseudo_data).flatten()
            diff = np.linalg.norm(self.b_ - b_new)/np.linalg.norm(b_new)
            self.diffs.append(diff)
            self.b_ = b_new # update

            # check convergence
            if diff < self.tol:
                self.edof_ = self.estimate_edof_(bases, inner, BW)
                self.aic_ = self.estimate_AIC_(X, y, mu)
                self.aicc_ = self.estimate_AICc_(X, y, mu)
                return

        print 'did not converge'

    def fit(self, X, y):
        # Setup
        y = np.ravel(y)
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

    def estimate_edof_(self, bases=None, inner=None, BW=None, inner_BW=None, limit=50000):
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

            if inner_BW is None:
                return scale * bases.dot(inner).tocsr()[idxs[:max_]].T.multiply(BW[:,idxs[:max_]]).sum()
            else:
                return scale * BW[:,idxs[:max_]].multiply(inner_BW[:,idxs[:max_]]).sum()
        else:
            # no subsampling
            if inner_BW is None:
                return bases.dot(inner).T.multiply(BW).sum()
            else:
                return BW.multiply(inner_BW).sum()

    def estimate_AIC_(self, y=None, mu=None):
        """
        Akaike Information Criterion
        """
        estimated_scale = not(self.distribution.name in ['binomial', 'poisson']) # if we estimate the scale, that adds 2 dof
        return -2*self.loglikelihood_(y=y, mu=mu) + 2*self.edof_ + 2*estimated_scale

    def estimate_AICc_(self, X=None, y=None, mu=None):
        """
        corrected Akaike Information Criterion
        """
        if self.aic_ is None:
            self.aic_ = self.estimate_AIC_(X, y, mu)
        return self.aic_ + 2*(self.edof_ + 1)*(self.edof_ + 2)/(y.shape[0] - self.edof_ -2)

    def estimate_r2_(self, X=None, y=None, mu=None):
        """
        estimate some pseudo R^2 values
        """
        if mu is None:
            mu = self.glm_mu_(X=X)

        n = len(y)
        null_mu = y.mean() * np.ones_like(y)

        null_l = self.loglikelihood_(y=y, mu=null_mu)
        full_l = self.loglikelihood_(y=y, mu=mu)

        self.r2_mcfadden_ = 1. - full_l/null_l
        self.r2_mcfadden_adj_ = 1. - (full_l-self.edof_)/null_l
        self.r2_cox_snell_ = (1. - np.exp(2./n * (null_l - full_l)))
        self.r2_nagelkerke_ = self.r2_cox_snell_ / (1. - np.exp(2./n * null_l))

    def estimate_GCV_UBRE_(self, X=None, y=None, bases=None, gamma=10., add_scale=True):
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

        if bases is None:
            bases = self.bases_(X)
        lp = self.linear_predictor_(bases=bases)
        mu = self.link.mu(lp, self.distribution)
        n = y.shape[0]
        if self.distribution.name in ['binomial', 'poisson']:
            # scale is known, use UBRE
            scale = self.distribution.scale
            self.ubre_ = 1./n * self.distribution.deviance(mu=mu, y=y) - (~add_scale)*(scale) + 2.*gamma/n * self.edof_ * scale
        # scale unkown, use GCV
        self.gcv_ = (n * self.distribution.deviance(mu=mu, y=y)) / (n - gamma * self.edof_)**2

    def prediction_intervals(self, X, width=.95, quantiles=None):
        return self.get_quantiles_(X, width, quantiles, prediction=True)

    def confidence_intervals(self, X, width=.95, quantiles=None):
        return self.get_quantiles_(X, width, quantiles, prediction=False)

    def get_quantiles_(self, X, width, quantiles, B=None, lp=None, prediction=False, xform=True, feature=-1):
        if quantiles is not None:
            if issubclass(quantiles.__class__, (np.int, np.float)):
                quantiles = [quantiles]
        else:
            alpha = (1 - width)/2.
            quantiles = [alpha, 1 - alpha]
        for quantile in quantiles:
            assert (quantile**2 <= 1.), 'quantiles must be in [0, 1]'

        if B is None:
            B = self.bases_(X, feature=feature)
        if lp is None:
            lp = self.linear_predictor_(bases=B, feature=feature)
        idxs = self.select_feature_(feature)
        cov = self.cov_[idxs][:,idxs]

        scale = self.distribution.scale
        var = (B.dot(cov) * B.todense().A).sum(axis=1) * scale
        if prediction:
            var += scale

        lines = []
        for quantile in quantiles:
            t = sp.stats.t.ppf(quantile, df=self.edof_)
            lines.append(lp + t * var**0.5)

        if xform:
            return self.link.mu(np.vstack(lines).T, self.distribution)
        return np.vstack(lines).T

    def select_feature_(self, i):
        """
        tool for indexing by feature function.

        many coefficients and parameters are organized by feature.
        this tool returns all of the indices for a given feature.

        GAM intercept is considered the 0th feature.
        """
        assert i < len(self.n_bases_), 'out of range'
        assert i >=-1, 'out of range'

        if i == -1:
            # special case for selecting all features
            return np.arange(np.sum(self.n_bases_), dtype=int)

        a = np.sum(self.n_bases_[:i])
        b = np.sum(self.n_bases_[i])
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
            B = self.bases_(X, feature=i)
            lp = self.linear_predictor_(bases=B, feature=i)
            p_deps.append(lp)
            conf_intervals.append(self.get_quantiles_(X, width=width,
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
        #
        # self.distribution = DISTRIBUTIONS['binomial']()
        # self.link = LINK_FUNCTIONS['logit']()

      super(LogisticGAM, self).__init__(levels=1, distribution='binomial', link='logit', **kwargs)
      self._exclude += ['distribution', 'link']

    def accuracy(self, X=None, y=None, proba=None):
        if proba is None:
            proba = self.predict_mu(X)
        return ((proba > 0.5).astype(int) == y.ravel()).mean()

    def predict(self, X):
        return self.predict_mu(X) > 0.5

    def predict_proba(self, X):
        return self.predict_mu(X)
