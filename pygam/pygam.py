# -*- coding: utf-8 -*-

from __future__ import division
from collections import defaultdict, OrderedDict
from copy import deepcopy

import numpy as np
import scipy as sp
from scipy import stats

from core import Core
from penalties import cont_P, cat_P, wrap_penalty
from distributions import Distribution, NormalDist, BinomialDist
from links import Link, IdentityLink, LogitLink
from callbacks import CallBack, Deviance, Diffs, Accuracy, validate_callback
from utils import check_dtype, check_y, check_X, check_X_y, print_data, gen_edge_knots, b_spline_basis, combine


EPS = np.finfo(np.float64).eps # machine epsilon


DISTRIBUTIONS = {'normal': NormalDist,
                 'poisson': None,
                 'binomial': BinomialDist,
                 'gamma': None,
                 'inv_gaussian': None
                 }

LINK_FUNCTIONS = {'identity': IdentityLink,
                  'log': None,
                  'logit': LogitLink,
                  'inverse': None,
                  'inv_squared': None
                  }

CALLBACKS = {'deviance': Deviance,
             'diffs': Diffs,
             'accuracy': Accuracy
            }

DTYPES = {'float': np.float,
          'int': np.int}


class GAM(Core):
    """
    base Generalized Additive Model
    """
    def __init__(self, lam=0.6, n_iter=100, n_splines=25, spline_order=3,
                 penalty_matrix='auto', tol=1e-4, distribution='normal',
                 link='identity', callbacks=['deviance', 'diffs'],
                 fit_intercept=True, fit_linear=True, fit_splines=True,
                 dtype='auto'):

        self.n_iter = n_iter
        self.tol = tol
        self.lam = lam
        self.n_splines = n_splines
        self.spline_order = spline_order
        self.penalty_matrix = penalty_matrix
        self.distribution = distribution
        self.link = link
        self.callbacks = callbacks
        self.fit_intercept = fit_intercept
        self.fit_linear = fit_linear
        self.fit_splines = fit_splines
        self.dtype = dtype

        # created by other methods
        self._b = None # model coefficients
        self._n_coeffs = [] # useful for indexing into model coefficients
        self._edge_knots = []
        self._lam = []
        self._n_splines = []
        self._spline_order = []
        self._penalty_matrix = []
        self._dtype = []
        self._fit_linear = []
        self._fit_splines = []
        self._fit_intercept = None
        self._opt = 0 # use 0 for numerically stable optimizer, 1 for naive

        # statistics and logging
        self._statistics = None # dict of statistics
        self.logs = defaultdict(list)

        # exclude some variables
        super(GAM, self).__init__()
        self._exclude += ['logs']

    def _expand_attr(self, attr, n, dt_alt=None, msg=None):
        """
        tool to parse and duplicate initialization arguments
          into model parameters.
        typically we use this tool to take a single attribute like:
          self.lam = 0.6
        and make one copy per feature, ie:
          self._lam = [0.6, 0.6, 0.6]
        for a model with 3 features.

        if self.attr is an iterable of values of length n,
          then copy it verbatim to self._attr.
        otherwise extend the single value to a list of length n,
          and copy that to self._attr

        dt_alt is an alternative value for dtypes of type integer (ie discrete).
        so if our 3-feature dataset is of types [float, float, int],
          we could use this method to turn
            self.lam = 0.6
          into
            self.lam = [0.6, 0.6, 0.3]
        by calling
          self._expand_attr('lam', 3, dt_alt=0.3)

        Parameters
        ----------
        attr : string
          name of the attribute to expand
        n : int
          number of time to repeat the attribute
        dt_alt : object, deafult: None
          object to subsitute attribute for discrete features.
          if dt_alt is None, int features are treated the same as float features.
        msg: string, default: None
          custom error message to report if
            self.attr is iterable BUT len(self.attr) != n
          if msg is None, default message is used:
            'expected "attr" to have length X.shape[1], but found {}'.format(len(self.attr))

        Returns
        -------
        None
        """
        data = getattr(self, attr)

        _attr = '_' + attr
        if hasattr(data, '__iter__'):
            if not (len(data) == n):
                if msg is None:
                    msg = 'expected {} to have length X.shape[1], '\
                          'but found {}'.format(attr, len(data))
                raise ValueError(msg)
        else:
            data = [data] * n

        if dt_alt is not None:
            data = [d if dt != np.int else dt_alt for d,dt in zip(data, self._dtype)]

        setattr(self, _attr, data)

    def _validate_parameters(self):
        """
        method to sanitize model parameters
        """
        # fit_intercep
        if not isinstance(self.fit_intercept, bool):
            raise ValueError('fit_intercept must be type bool, but found {}'\
                             .format(self.fit_intercept.__class__))

        # n_iter
        if not ((self.n_iter >= 1) and isinstance(self.n_iter, int)):
            raise ValueError('n_iter must be int >= 1. found n_iter = {}'\
                             .format(self.n_iter))

        # lam
        if (np.array(self.lam).astype(float) != np.array(self.lam)).all() or \
           np.array(self.lam) <= 0:
            raise ValueError("lam must be in float > 0, "\
                             "or iterable of floats > 0, "\
                             "but found lam = {}".format(self.lam))

        # n_splines
        if (np.array(self.n_splines).astype(int) != np.array(self.n_splines)).all() or \
           np.array(self.n_splines) < 0:
            raise ValueError("n_splines must be in int >= 0, "\
                             "or iterable of ints >= 0, "\
                             "but found n_splines = {}".format(self.n_splines))

        # spline_order
        if (np.array(self.spline_order).astype(int) != np.array(self.spline_order)).all() or \
           np.array(self.spline_order) < 0:
            raise ValueError("spline_order must be in int >= 0, "\
                             "or iterable of ints >= 0, "\
                             "but found spline_order = {}".format(self.spline_order))

        # n_splines + spline_order
        if not (np.atleast_1d(self.n_splines) > np.atleast_1d(self.spline_order)).all():
            raise ValueError('n_splines must be > spline_order. '\
                             'found: n_splines = {} and spline_order = {}'\
                             .format(self.n_splines, self.spline_order))

        # distribution
        if not ((self.distribution in DISTRIBUTIONS)
                or isinstance(self.distribution, Distribution)):
            raise ValueError('unsupported distribution {}'.format(self.distribution))
        self.distribution = DISTRIBUTIONS[self.distribution]() if self.distribution in DISTRIBUTIONS else self.distribution

        # link
        if not ((self.link in LINK_FUNCTIONS) or isinstance(self.link, Link)):
            raise ValueError('unsupported link {}'.format(self.link))
        self.link = LINK_FUNCTIONS[self.link]() if self.link in LINK_FUNCTIONS else self.link

        # callbacks
        if not hasattr(self.callbacks, '__iter__'):
            raise ValueError('callbacks must be iterable. found {}'\
                             .format(self.callbacks))

        if not all([c in ['deviance', 'diffs', 'accuracy']
                    or isinstance(c, CallBack) for c in self.callbacks]):
            raise ValueError('unsupported callback(s) {}'.format(self.callbacks))
        self.callbacks = [CALLBACKS[c]() if (c in CALLBACKS) else c for c in self.callbacks]
        self.callbacks = [validate_callback(c) for c in self.callbacks]

        # penalty_matrix
        if not (hasattr(self.penalty_matrix, '__iter__') or
                callable(self.penalty_matrix) or
                self.penalty_matrix=='auto'):
            raise ValueError('penalty_matrix must be iterable or callable, '\
                             'but found {}'.format(self.penalty_matrix))
        if hasattr(self.penalty_matrix, '__iter__'):
            for i, pmat in enumerate(self.penalty_matrix):
                if not (callable(pmat) or pmat=='auto'):
                    raise ValueError('penalty_matrix must be callable or "auto", '\
                                     'but found {} for {}th penalty'.format(pmat, i))

        # dtype
        if not (self.dtype in ['auto', 'float', 'int'] or
                hasattr(self.dtype, '__iter__')):
            raise ValueError("dtype must be in ['auto', 'float', 'int'] or "\
                             "iterable of those strings, "\
                             "but found dtype = {}".format(self.dtype))
        if hasattr(self.dtype, '__iter__'):
            for dt in self.dtype:
                if dt not in ['auto', 'float', 'int']:
                    raise ValueError("elements of iterable dtype must be in "\
                                     "['auto', 'float', 'int], but found "\
                                     "dtype = {}".format(self.dtype))

    def _validate_data_dep_params(self, X):
        """
        method to validate and prepare data-dependent parameters
        """
        n_samples, n_features = X.shape

        # set up dtypes and check types if 'auto'
        self._expand_attr('dtype', n_features)
        for i, (dt, x) in enumerate(zip(self._dtype, X.T)):
            if dt == 'auto':
                dt = check_dtype(x)[0]
            else:
                dt = DTYPES[dt]
            self._dtype[i] = dt
        assert len(self._dtype) == n_features # sanity check

        # set up lambdas
        self._expand_attr('lam', n_features)
        if self.fit_intercept:
            self._lam = [0.] + self._lam # add intercept term

        # set up penalty matrices
        self._expand_attr('penalty_matrix', n_features)

        # set up fit_linear and fit_splines, copy fit_intercept
        self._fit_intercept = self.fit_intercept
        self._expand_attr('fit_linear', n_features, dt_alt=False)
        self._expand_attr('fit_splines', n_features)
        line_or_spline = [bool(line + spline) for line, spline in zip(self._fit_linear, self._fit_splines)]
        if not all(line_or_spline):
            raise ValueError('a line or a spline must be fit on each feature. '\
                             'Neither were found on feature(s): {}' \
                             .format([i for i, T in enumerate(line_or_spline) if not T ]))

        # expand spline_order, n_splines, and prepare edge_knots
        self._expand_attr('spline_order', X.shape[1], dt_alt=0)
        self._expand_attr('n_splines', X.shape[1], dt_alt=0)
        self._edge_knots = [gen_edge_knots(feat, dtype) for feat, dtype in zip(X.T, self._dtype)]

        # update our n_splines correcting for categorical features, no splines
        for i, (fs, dt, ek) in enumerate(zip(self._fit_splines,
                                             self._dtype,
                                             self._edge_knots)):
            if fs:
                if dt == np.int:
                    self._n_splines[i] = len(ek) - 1
            if not fs:
                self._n_splines[i] = 0

        # compute number of model coefficients
        self._n_coeffs = []
        for n_splines, fit_linear, fit_splines in zip(self._n_splines,
                                                      self._fit_linear,
                                                      self._fit_splines):
            self._n_coeffs.append(n_splines * fit_splines + fit_linear)
        if self._fit_intercept:
            self._n_coeffs = [1] + self._n_coeffs

        # check enough data
        if sum(self._n_coeffs) > n_samples:
            raise ValueError('Require num samples >= num model coefficients. '\
                             'Model has a total of {} coefficients, but only '\
                             'found {} samples.'.format(sum(self._n_coeffs),
                                                        n_samples))

    def _loglikelihood(self, y, mu):
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
        if feature >= len(self._n_coeffs) or feature < -1:
            raise ValueError('feature {} out of range for X with shape {}'\
                             .format(feature, X.shape))

        # for all features, build matrix recursively
        if feature == -1:
            modelmat = []
            for feat in range(X.shape[1] + self._fit_intercept):
                modelmat.append(self._modelmat(X, feature=feat))
            return sp.sparse.hstack(modelmat, format='csc')

        # intercept
        if (feature == 0) and self._fit_intercept:
            return sp.sparse.csc_matrix(np.ones((X.shape[0], 1)))

        # return only the basis functions for 1 feature
        feature = feature - self._fit_intercept
        featuremat = []
        if self._fit_linear[feature]:
            featuremat.append(sp.sparse.csc_matrix(X[:, feature][:,None]))
        if self._fit_splines[feature]:
            featuremat.append(b_spline_basis(X[:,feature],
                                             edge_knots=self._edge_knots[feature],
                                             spline_order=self._spline_order[feature],
                                             n_splines=self._n_splines[feature],
                                             sparse=True))

        return sp.sparse.hstack(featuremat, format='csc')

    def _P(self):
        """
        penatly matrix for P-Splines

        builds the GLM block-diagonal penalty matrix out of
        proto-penalty matrices from each feature.

        each proto-penalty matrix is multiplied by a lambda for that feature.
        the first feature is the intercept.

        so for m features:
        P = block_diag[lam0 * P0, lam1 * P1, lam2 * P2, ... , lamm * Pm]
        """
        Ps = []

        if self._fit_intercept:
            Ps.append(np.array(1))

        for n, fit_linear, dtype, pmat in zip(self._n_coeffs[self._fit_intercept:],
                                              self._fit_linear,
                                              self._dtype,
                                              self._penalty_matrix):
            if pmat in ['auto', None]:
                if dtype == np.float:
                    p = cont_P
                if dtype == np.int:
                    p = cat_P
            Ps.append(wrap_penalty(p, fit_linear)(n))

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
            self._b = np.zeros(m) # allow more training

        P = self._P() # create penalty matrix
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

        # estimate statistics even if not converged
        self._estimate_model_statistics(Y, modelmat, inner=None, BW=WB.T, B=B)
        if diff < self.tol:
            return

        print 'did not converge'
        return

    def _pirls_naive(self, X, y):
        modelmat = self._modelmat(X) # build a basis matrix for the GLM
        m = modelmat.shape[1]

        # initialize GLM coefficients
        if self._b is None:
            self._b = np.zeros(m) # allow more training

        P = self._P() # create penalty matrix
        P += sp.sparse.diags(np.ones(m) * np.sqrt(EPS)) # improve condition

        for _ in range(self.n_iter):
            lp = self._linear_predictor(modelmat=modelmat)
            mu = self.link.mu(lp, self.distribution)

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

        # validate parameters
        self._validate_parameters()

        # validate data
        y = check_y(y, self.link, self.distribution)
        X = check_X(X)
        check_X_y(X, y)

        # validate data-dependent parameters
        self._validate_data_dep_params(X)

        # optimize
        if self._opt == 0:
            self._pirls(X, y)
        if self._opt == 1:
            self._pirls_naive(X, y)
        return self

    def deviance_residuals(self, X, y, scaled=False):
        """
        method to compute the deviance residuals of the model

        these are analogous to the residuals of an OLS.

        Parameters
        ----------
        X : array-like
          input data array of shape (n_saples, n_features)
        y : array-like
          output data vector of shape (n_samples,)
        scaled : bool, default: False
          whether to scale the deviance by the (estimated) distribution scale

        Returns
        -------
        deviance_residuals : np.array
          with shape (n_samples,)
        """
        mu = self.predict_mu(X)
        sign = np.sign(y-mu)
        return sign * self.distribution.deviance(y, mu, summed=False, scaled=scaled)**0.5

    def _estimate_model_statistics(self, y, modelmat, inner=None, BW=None, B=None):
        """
        method to compute all of the model statistics
        """
        self._statistics = {}

        lp = self._linear_predictor(modelmat=modelmat)
        mu = self.link.mu(lp, self.distribution)
        self._statistics['edof'] = self._estimate_edof(BW=BW, B=B)
        # self.edof_ = np.dot(U1, U1.T).trace().A.flatten() # this is wrong?
        if not self.distribution._known_scale:
            self.distribution.scale = self.distribution.phi(y=y, mu=mu, edof=self._statistics['edof'])
        self._statistics['scale'] = self.distribution.scale
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
        estimated_scale = not(self.distribution._known_scale) # if we estimate the scale, that adds 2 dof
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
            mu = self.predict_mu_(X=X)

        n = len(y)
        null_mu = y.mean() * np.ones_like(y)

        null_d = self.distribution.deviance(y=y, mu=null_mu)
        full_d = self.distribution.deviance(y=y, mu=mu)

        r2 = OrderedDict()
        r2['explained_deviance'] = 1. - full_d/null_d
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
        if gamma < 1:
            raise ValueError('gamma scaling should be greater than 1, '\
                             'but found gama = {}',format(gamma))

        if modelmat is None:
            modelmat = self._modelmat(X)

        lp = self._linear_predictor(modelmat=modelmat)
        mu = self.link.mu(lp, self.distribution)
        n = y.shape[0]
        edof = self._statistics['edof']

        GCV = None
        UBRE = None

        if self.distribution._known_scale:
            # scale is known, use UBRE
            scale = self.distribution.scale
            UBRE = 1./n * self.distribution.deviance(mu=mu, y=y, scaled=False) - (~add_scale)*(scale) + 2.*gamma/n * edof * scale
        else:
            # scale unkown, use GCV
            GCV = (n * self.distribution.deviance(mu=mu, y=y, scaled=False)) / (n - gamma * edof)**2
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
            if (quantile > 1) or (quantile < 0):
                raise ValueError('quantiles must be in [0, 1], but found {}'\
                                 .format(quantiles))

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

    def _select_feature(self, feature):
        """
        tool for indexing by feature function.

        many coefficients and parameters are organized by feature.
        this tool returns all of the indices for a given feature.

        GAM intercept is considered the 0th feature.
        """
        if feature >= len(self._n_coeffs) or feature < -1:
            raise ValueError('feature {} out of range for X with shape {}'\
                             .format(feature, X.shape))

        if feature == -1:
            # special case for selecting all features
            return np.arange(np.sum(self._n_coeffs), dtype=int)

        a = np.sum(self._n_coeffs[:feature])
        b = np.sum(self._n_coeffs[feature])
        return np.arange(a, a+b, dtype=int)

    def partial_dependence(self, X, features=None, width=None, quantiles=None):
        """
        Computes the feature functions for the GAM as well as their confidence intervals.
        """
        m = len(self._n_coeffs) - self._fit_intercept
        p_deps = []

        compute_quantiles = (width is not None) or (quantiles is not None)
        conf_intervals = []

        if features is None:
            features = np.arange(m) + self._fit_intercept

        # convert to array
        features = np.atleast_1d(features)

        # ensure feature exists
        if (features >= len(self._n_coeffs)).any() or (features < -1).any():
            raise ValueError('features {} out of range for X with shape {}'\
                             .format(features, X.shape))

        for i in features:
            B = self._modelmat(X, feature=i)
            lp = self._linear_predictor(modelmat=B, feature=i)
            p_deps.append(lp)

            if compute_quantiles:
                conf_intervals.append(self._get_quantiles(X, width=width,
                                                          quantiles=quantiles,
                                                          B=B, lp=lp,
                                                          feature=i, xform=False))
        if compute_quantiles:
            return np.vstack(p_deps).T, conf_intervals
        return np.vstack(p_deps).T

    def summary(self):
        """
        produce a summary of the model statistics

        #TODO including feature significance via F-Test
        """
        assert bool(self._statistics), 'GAM has not been fitted'

        keys = ['edof', 'AIC', 'AICc']
        if self.distribution._known_scale:
            keys.append('UBRE')
        else:
            keys.append('GCV')
        keys.append('scale')

        sub_data = OrderedDict([[k, self._statistics[k]] for k in keys])

        print_data(sub_data, title='Model Statistics')
        print('')
        print_data(self._statistics['pseudo_r2'], title='Pseudo-R^2')

    def gridsearch(self, X, y, return_scores=False, keep_best=True,
                   objective='auto', **param_grids):
        """
        grid search method

        search for the GAM with the lowest GCV/UBRE score across 1 lambda
        or multiple lambas.

        NOTE:
        gridsearch method is lazy and will not remove useless combinations
        from the search space, eg.
          n_splines=np.arange(5,10), fit_splines=[True, False]
        will result in 10 loops, of which 5 are equivalent because fit_splines==False

        it is not recommended to search over a grid that alternates
        between known scales and unknown scales, as the scores of the
        cadidate models will not be comparable.

        Parameters
        ----------
        X : array
          input data of shape (n_samples, m_features)

        y : array
          label data of shape (n_samples,)

        return_scores : boolean, default False
          whether to return the hyperpamaters and score for each element in the grid

        keep_best : boolean
          whether to keep the best GAM as self.
          default: True

        objective : string, default: 'auto'
          metric to optimize. must be in ['AIC', 'AICc', 'GCV', 'UBRE', 'auto']
          if 'auto', then grid search will optimize GCV for models with unknown
          scale and UBRE for models with known scale.

        **kwargs : dict, default {'lam': np.logspace(-3, 3, 11)}
          pairs of parameters and iterables of floats, or
          parameters and iterables of iterables of floats.

          if iterable of iterables of floats, the outer iterable must have
          length m_features.

          the method will make a grid of all the combinations of the parameters
          and fit a GAM to each combination.


        Returns
        -------
        if return_values == True:
            model_scores : dict
              Contains each fitted model as keys and corresponding
              GCV/UBRE scores as values
        else:
            self, ie possible the newly fitted model
        """
        # validate objective
        if objective not in ['auto', 'GCV', 'UBRE', 'AIC', 'AICc']:
            raise ValueError("objective mut be in "\
                             "['auto', 'GCV', 'UBRE', 'AIC', 'AICc'], but found "\
                             "objective = {}".format(objective))

        if objective == 'GCV' and self.distribution._known_scale:
            raise ValueError('GCV should be used for models with unknown scale')
        if objective == 'UBRE' and not self.distribution._known_scale:
            raise ValueError('UBRE should be used for models with known scale')

        # default gridsearch
        if not bool(param_grids):
            param_grids['lam'] = np.logspace(-3, 3, 11)

        admissible_params = self.get_params()
        params = []
        grids = []
        for param, grid in param_grids.iteritems():
            if param not in (admissible_params):
                raise ValueError('unknown parameter {}'.format(param))
            if not (hasattr(grid, '__iter__') and (len(grid) > 1)): \
                raise ValueError('{} grid must either be iterable of iterables, '\
                                 'or an iterable of lengnth > 1, but found {}'\
                                 .format(param, grid))

            # prepare grid
            if any(hasattr(g, '__iter__') for g in grid):
                # cast to np.array
                grid = [np.atleast_1d(g) for g in grid]
                # set grid to combination of all grids
                grid = combine(*grid)
            else:
                grid = grid

            # save param name and grid
            params.append(param)
            grids.append(grid)

        # build a list of dicts of candidate model params
        param_grid_list = []
        for candidate in combine(*grids):
            param_grid_list.append(dict(zip(params,candidate)))

        # set up data collection
        best_model = None # keep the best model
        best_score = np.inf
        scores = []
        models = []
        if objective == 'auto':
            if self.distribution._known_scale:
                objective = 'UBRE'
            else:
                objective = 'GCV'

        # check if our model has been fitted already
        if self._statistics is not None:
            models.append(self)
            scores.append(self._statistics[objective])

            # our model is currently the best
            best_model = models[-1]
            best_score = scores[-1]

        # loop through candidate model params
        for param_grid in param_grid_list:
            # train new model
            gam = eval(repr(self))
            gam.set_params(**param_grid)
            try:
                gam.fit(X, y)
            except ValueError as error:
                print str(error)
                print 'on model:'
                print str(gam)
                print 'skipping...\n'
                continue

            # record results
            models.append(gam)
            scores.append(gam._statistics[objective])

            # track best
            if scores[-1] < best_score:
                best_model = models[-1]
                best_score = scores[-1]

        if len(models) == 0:
            print 'no models were ffitted'
            return self
        if keep_best:
            self.set_params(deep=True, **best_model.get_params(deep=True))
        if return_scores:
            return OrderedDict(zip(models, scores))
        else:
            return self


class LinearGAM(GAM):
    """
    Linear GAM model
    """
    def __init__(self, lam=0.6, n_iter=100, n_splines=25, spline_order=3,
                 penalty_matrix='auto', dtype='auto', tol=1e-4, scale=None,
                 callbacks=['deviance', 'diffs'],
                 fit_intercept=True, fit_linear=True, fit_splines=True):
        self.scale = scale
        super(LinearGAM, self).__init__(distribution=NormalDist(scale=self.scale),
                                        link='identity',
                                        lam=lam,
                                        dtype=dtype,
                                        n_iter=n_iter,
                                        n_splines=n_splines,
                                        spline_order=spline_order,
                                        penalty_matrix=penalty_matrix,
                                        tol=tol,
                                        callbacks=callbacks,
                                        fit_intercept=fit_intercept,
                                        fit_linear=fit_linear,
                                        fit_splines=fit_splines)

        self._exclude += ['distribution', 'link']

class LogisticGAM(GAM):
    """
    Logistic GAM model
    """
    def __init__(self, lam=0.6, n_iter=100, n_splines=25, spline_order=3,
                 penalty_matrix='auto', dtype='auto', tol=1e-4,
                 callbacks=['deviance', 'diffs', 'accuracy'],
                 fit_intercept=True, fit_linear=True, fit_splines=True):
        super(LogisticGAM, self).__init__(distribution='binomial',
                                          link='logit',
                                          lam=lam,
                                          dtype=dtype,
                                          n_iter=n_iter,
                                          n_splines=n_splines,
                                          spline_order=spline_order,
                                          penalty_matrix=penalty_matrix,
                                          tol=tol,
                                          callbacks=callbacks,
                                          fit_intercept=fit_intercept,
                                          fit_linear=fit_linear,
                                          fit_splines=fit_splines)

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
