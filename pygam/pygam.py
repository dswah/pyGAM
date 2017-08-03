# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from collections import defaultdict
from collections import OrderedDict
from copy import deepcopy
from progressbar import ProgressBar
import warnings

import numpy as np
import scipy as sp
from scipy import stats

from pygam.core import Core

from pygam.penalties import derivative
from pygam.penalties import l2
from pygam.penalties import monotonic_inc
from pygam.penalties import monotonic_dec
from pygam.penalties import convex
from pygam.penalties import concave
from pygam.penalties import circular
from pygam.penalties import none
from pygam.penalties import wrap_penalty

from pygam.distributions import Distribution
from pygam.distributions import NormalDist
from pygam.distributions import BinomialDist
from pygam.distributions import PoissonDist
from pygam.distributions import GammaDist
from pygam.distributions import InvGaussDist

from pygam.links import Link
from pygam.links import IdentityLink
from pygam.links import LogitLink
from pygam.links import LogLink
from pygam.links import InverseLink
from pygam.links import InvSquaredLink

from pygam.callbacks import CallBack
from pygam.callbacks import Deviance
from pygam.callbacks import Diffs
from pygam.callbacks import Accuracy
from pygam.callbacks import Coef
from pygam.callbacks import validate_callback

from pygam.utils import check_dtype
from pygam.utils import check_y
from pygam.utils import check_X
from pygam.utils import check_X_y
from pygam.utils import check_lengths
from pygam.utils import print_data
from pygam.utils import gen_edge_knots
from pygam.utils import b_spline_basis
from pygam.utils import combine
from pygam.utils import cholesky
from pygam.utils import check_param
from pygam.utils import isiterable
from pygam.utils import NotPositiveDefiniteError


EPS = np.finfo(np.float64).eps # machine epsilon


DISTRIBUTIONS = {'normal': NormalDist,
                 'poisson': PoissonDist,
                 'binomial': BinomialDist,
                 'gamma': GammaDist,
                 'inv_gauss': InvGaussDist
                 }

LINK_FUNCTIONS = {'identity': IdentityLink,
                  'log': LogLink,
                  'logit': LogitLink,
                  'inverse': InverseLink,
                  'inv_squared': InvSquaredLink
                  }

CALLBACKS = {'deviance': Deviance,
             'diffs': Diffs,
             'accuracy': Accuracy,
             'coef': Coef
            }

PENALTIES = {'auto': 'auto',
             'derivative': derivative,
             'l2': l2,
             'none': none,
            }

CONSTRAINTS = {'convex': convex,
               'concave': concave,
               'monotonic_inc': monotonic_inc,
               'monotonic_dec': monotonic_dec,
               'circular': circular,
               'none': none
              }


class GAM(Core):
    """Generalized Additive Model

    Parameters
    ----------
    callbacks : list of strings or list of CallBack objects,
                default: ['deviance', 'diffs']
        Names of callback objects to call during the optimization loop.

    constraints : str or callable, or iterable of str or callable,
                  default: None
        Names of constraint functions to call during the optimization loop.

        Must be in {'convex', 'concave', 'monotonic_inc', 'monotonic_dec',
                    'circular', 'none'}

        If None, then the model will apply no constraints.

        If only one str or callable is specified, then is it copied for all
        features.

    distribution : str or Distribution object, default: 'normal'
        Distribution to use in the model.

    link : str or Link object, default: 'identity'
        Link function to use in the model.

    dtype : str in {'auto', 'numerical',  'categorical'},
            or list of str, default: 'auto'
        String describing the data-type of each feature.

        'numerical' is used for continuous-valued data-types,
            like in regression.
        'categorical' is used for discrete-valued data-types,
            like in classification.

        If only one str is specified, then is is copied for all features.

    lam : float or iterable of floats > 0, default: 0.6
        Smoothing strength; must be a positive float, or one positive float
        per feature.

        Larger values enforce stronger smoothing.

        If only one float is specified, then it is copied for all features.

    fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

        NOTE: the intercept receives no smoothing penalty.

    fit_linear : bool or iterable of bools, default: False
        Specifies if a linear term should be added to any of the feature
        functions. Useful for including pre-defined feature transformations
        in the model.

        If only one bool is specified, then it is copied for all features.

        NOTE: Many constraints are incompatible with an additional linear fit.
            eg. if a non-zero linear function is added to a periodic spline
            function, it will cease to be periodic.

            this is also possible for a monotonic spline function.

    fit_splines : bool or iterable of bools, default: True
        Specifies if a smoother should be added to any of the feature
        functions. Useful for defining feature transformations a-priori
        that should not have splines fitted to them.

        If only one bool is specified, then it is copied for all features.

        NOTE: fit_splines supercedes n_splines.
        ie. if n_splines > 0 and fit_splines = False, no splines will be fitted.

    max_iter : int, default: 100
        Maximum number of iterations allowed for the solver to converge.

    penalties : str or callable, or iterable of str or callable,
                default: 'auto'
        Type of penalty to use for each feature.

        penalty should be in {'auto', 'none', 'derivative', 'l2', }

        If 'auto', then the model will use 2nd derivative smoothing for features
        of dtype 'numerical', and L2 smoothing for features of dtype
        'categorical'.

        If only one str or callable is specified, then is it copied for all
        features.

    n_splines : int, or iterable of ints, default: 25
        Number of splines to use in each feature function; must be non-negative.
        If only one int is specified, then it is copied for all features.

        Note: this value is set to 0 if fit_splines is False

    spline_order : int, or iterable of ints, default: 3
        Order of spline to use in each feature function; must be non-negative.
        If only one int is specified, then it is copied for all features

        Note: if a feature is of type categorical, spline_order will be set to 0.

    tol : float, default: 1e-4
        Tolerance for stopping criteria.

    Attributes
    ----------
    coef_ : array, shape (n_classes, m_features)
        Coefficient of the features in the decision function.
        If fit_intercept is True, then self.coef_[0] will contain the bias.

    statistics_ : dict
        Dictionary containing model statistics like GCV/UBRE scores, AIC/c,
        parameter covariances, estimated degrees of freedom, etc.

    logs_ : dict
        Dictionary containing the outputs of any callbacks at each
        optimization loop.

        The logs are structured as `{callback: [...]}`

    References
    ----------
    Simon N. Wood, 2006
    Generalized Additive Models: an introduction with R

    Hastie, Tibshirani, Friedman
    The Elements of Statistical Learning
    http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf

    Paul Eilers & Brian Marx, 2015
    International Biometric Society: A Crash Course on P-splines
    http://www.ibschannel2015.nl/project/userfiles/Crash_course_handout.pdf
    """
    def __init__(self, lam=0.6, max_iter=100, n_splines=25, spline_order=3,
                 penalties='auto', tol=1e-4, distribution='normal',
                 link='identity', callbacks=['deviance', 'diffs'],
                 fit_intercept=True, fit_linear=False, fit_splines=True,
                 dtype='auto', constraints=None):

        self.max_iter = max_iter
        self.tol = tol
        self.lam = lam
        self.n_splines = n_splines
        self.spline_order = spline_order
        self.penalties = penalties
        self.distribution = distribution
        self.link = link
        self.callbacks = callbacks
        self.constraints = constraints
        self.fit_intercept = fit_intercept
        self.fit_linear = fit_linear
        self.fit_splines = fit_splines
        self.dtype = dtype

        # created by other methods
        self._n_coeffs = [] # useful for indexing into model coefficients
        self._edge_knots = []
        self._lam = []
        self._n_splines = []
        self._spline_order = []
        self._penalties = []
        self._constraints = []
        self._dtype = []
        self._fit_linear = []
        self._fit_splines = []
        self._fit_intercept = None

        # internal settings
        self._constraint_lam = 1e9 # regularization intensity for constraints
        self._constraint_l2 = 1e-3 # diagononal loading to improve conditioning
        self._constraint_l2_max = 1e-1 # maximum loading
        self._opt = 0 # use 0 for numerically stable optimizer, 1 for naive

        # call super and exclude any variables
        super(GAM, self).__init__()

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

        dt_alt is an alternative value for dtypes of type categorical (ie discrete).
        so if our 3-feature dataset is of types
            ['numerical', 'numerical', 'categorical'],
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
          object to subsitute attribute for categorical features.
          if dt_alt is None, categorical features are treated the same as
          numerical features.
        msg: string, default: None
          custom error message to report if
            self.attr is iterable BUT len(self.attr) != n
          if msg is None, default message is used:
            'expected "attr" to have length X.shape[1], but found {}'.format(len(self.attr))

        Returns
        -------
        None
        """
        data = deepcopy(getattr(self, attr))

        _attr = '_' + attr
        if isiterable(data):
            if not (len(data) == n):
                if msg is None:
                    msg = 'expected {} to have length X.shape[1], '\
                          'but found {}'.format(attr, len(data))
                raise ValueError(msg)
        else:
            data = [data] * n

        if dt_alt is not None:
            data = [d if dt != 'categorical' else dt_alt for d,dt in zip(data, self._dtype)]

        setattr(self, _attr, data)

    @property
    def _is_fitted(self):
        """
        simple way to check if the GAM has been fitted

        Parameters
        ---------
        None

        Returns
        -------
        bool : whether or not the model is fitted
        """
        return hasattr(self, 'coef_')

    def _validate_params(self):
        """
        method to sanitize model parameters

        Parameters
        ---------
        None

        Returns
        -------
        None
        """
        # fit_intercep
        if not isinstance(self.fit_intercept, bool):
            raise ValueError('fit_intercept must be type bool, but found {}'\
                             .format(self.fit_intercept.__class__))

        # max_iter
        self.max_iter = check_param(self.max_iter, param_name='max_iter',
                                    dtype='int', constraint='>=1',
                                    iterable=False)

        # lam
        self.lam = check_param(self.lam, param_name='lam',
                               dtype='float', constraint='>0')

        # n_splines
        self.n_splines = check_param(self.n_splines, param_name='n_splines',
                                     dtype='int', constraint='>=0')

        # spline_order
        self.spline_order = check_param(self.spline_order,
                                        param_name='spline_order',
                                        dtype='int', constraint='>=0')

        # n_splines + spline_order
        if not (np.atleast_1d(self.n_splines) >
                np.atleast_1d(self.spline_order)).all():
            raise ValueError('n_splines must be > spline_order. '\
                             'found: n_splines = {} and spline_order = {}'\
                             .format(self.n_splines, self.spline_order))

        # distribution
        if not ((self.distribution in DISTRIBUTIONS)
                or isinstance(self.distribution, Distribution)):
            raise ValueError('unsupported distribution {}'.format(self.distribution))
        if self.distribution in DISTRIBUTIONS:
            self.distribution = DISTRIBUTIONS[self.distribution]()

        # link
        if not ((self.link in LINK_FUNCTIONS) or isinstance(self.link, Link)):
            raise ValueError('unsupported link {}'.format(self.link))
        if self.link in LINK_FUNCTIONS:
            self.link = LINK_FUNCTIONS[self.link]()

        # callbacks
        if not isiterable(self.callbacks):
            raise ValueError('Callbacks must be iterable, but found {}'\
                             .format(self.callbacks))

        if not all([c in CALLBACKS or
                    isinstance(c, CallBack) for c in self.callbacks]):
            raise ValueError('unsupported callback(s) {}'.format(self.callbacks))
        callbacks = list(self.callbacks)
        for i, c in enumerate(self.callbacks):
            if c in CALLBACKS:
                callbacks[i] = CALLBACKS[c]()
        self.callbacks = [validate_callback(c) for c in callbacks]

        # penalties
        if not (isiterable(self.penalties) or
                hasattr(self.penalties, '__call__') or
                self.penalties in PENALTIES or
                self.penalties is None):
            raise ValueError('penalties must be iterable or callable, '\
                             'but found {}'.format(self.penalties))

        if isiterable(self.penalties):
            for i, p in enumerate(self.penalties):
                if not (hasattr(p, '__call__') or
                        (p in PENALTIES) or
                        (p is None)):
                    raise ValueError("penalties must be callable or in "\
                                     "{}, but found {} for {}th penalty"\
                                     .format(list(PENALTIES.keys()), p, i))

        # constraints
        if not (isiterable(self.constraints) or
                hasattr(self.constraints, '__call__') or
                self.constraints in CONSTRAINTS or
                self.constraints is None):
            raise ValueError('constraints must be iterable or callable, '\
                             'but found {}'.format(self.constraints))

        if isiterable(self.constraints):
            for i, c in enumerate(self.constraints):
                if not (hasattr(c, '__call__') or
                        (c in CONSTRAINTS) or
                        (c is None)):
                    raise ValueError("constraints must be callable or in "\
                                     "{}, but found {} for {}th constraint"\
                                     .format(list(CONSTRAINTS.keys()), c, i))

        # dtype
        if not (self.dtype in ['auto', 'numerical', 'categorical'] or
                isiterable(self.dtype)):
            raise ValueError("dtype must be in ['auto', 'numerical', "\
                             "'categorical'] or iterable of those strings, "\
                             "but found dtype = {}".format(self.dtype))

        if isiterable(self.dtype):
            for dt in self.dtype:
                if dt not in ['auto', 'numerical', 'categorical']:
                    raise ValueError("elements of iterable dtype must be in "\
                                     "['auto', 'numerical', 'categorical], "\
                                     "but found dtype = {}".format(self.dtype))

    def _validate_data_dep_params(self, X):
        """
        method to validate and prepare data-dependent parameters

        Parameters
        ---------
        X : array-like
            containing the input dataset

        Returns
        -------
        None
        """
        n_samples, m_features = X.shape

        # set up dtypes and check types if 'auto'
        self._expand_attr('dtype', m_features)
        for i, (dt, x) in enumerate(zip(self._dtype, X.T)):
            if dt == 'auto':
                dt = check_dtype(x)[0]
                if dt == 'categorical':
                    warnings.warn('detected catergorical data for feature {}'\
                                  .format(i), stacklevel=2)
            self._dtype[i] = dt
        assert len(self._dtype) == m_features # sanity check

        # set up lambdas
        self._expand_attr('lam', m_features)

        # add intercept term
        if self.fit_intercept:
            self._lam = [0.] + self._lam

        # set up penalty matrices
        self._expand_attr('penalties', m_features)

        # set up constraints
        self._expand_attr('constraints', m_features, dt_alt=None)

        # set up fit_linear and fit_splines, copy fit_intercept
        self._fit_intercept = self.fit_intercept
        self._expand_attr('fit_linear', m_features, dt_alt=False)
        self._expand_attr('fit_splines', m_features)
        for i, (fl, c) in enumerate(zip(self._fit_linear, self._constraints)):
            if bool(c) and (c is not 'none'):
                if fl:
                    warnings.warn('cannot do fit_linear with constraints. '\
                                  'setting fit_linear=False for feature {}'\
                                  .format(i))
                self._fit_linear[i] = False

        line_or_spline = [bool(line + spline) for line, spline in \
                          zip(self._fit_linear, self._fit_splines)]
        # problems
        if not all(line_or_spline):
            bad = [i for i, l_or_s in enumerate(line_or_spline) if not l_or_s]
            raise ValueError('a line or a spline must be fit on each feature. '\
                             'Neither were found on feature(s): {}' \
                             .format(bad))

        # expand spline_order, n_splines, and prepare edge_knots
        self._expand_attr('spline_order', X.shape[1], dt_alt=0)
        self._expand_attr('n_splines', X.shape[1], dt_alt=0)
        self._edge_knots = [gen_edge_knots(feat, dtype) for feat, dtype in \
                            zip(X.T, self._dtype)]

        # update our n_splines correcting for categorical features, no splines
        for i, (fs, dt, ek) in enumerate(zip(self._fit_splines,
                                             self._dtype,
                                             self._edge_knots)):
            if fs:
                if dt == 'categorical':
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

    def loglikelihood(self, X, y, weights=None):
        """
        compute the log-likelihood of the dataset using the current model

        Parameters
        ---------
        X : array-like of shape (n_samples, m_features)
            containing the input dataset
        y : array-like of shape (n,)
            containing target values
        weights : array-like of shape (n,)
            containing sample weights

        Returns
        -------
        log-likelihood : np.array of shape (n,)
            containing log-likelihood scores
        """
        mu = self.predict_mu(X)
        return self._loglikelihood(y, mu, weights=weights)

    def _loglikelihood(self, y, mu, weights=None):
        """
        compute the log-likelihood of the dataset using the current model

        Parameters
        ---------
        y : array-like of shape (n,)
            containing target values
        mu : array-like of shape (n_samples,)
            expected value of the targets given the model and inputs
        weights : array-like of shape (n,)
            containing sample weights

        Returns
        -------
        log-likelihood : np.array of shape (n,)
            containing log-likelihood scores
        """
        return np.log(self.distribution.pdf(y=y, mu=mu, weights=weights)).sum()

    def _linear_predictor(self, X=None, modelmat=None, b=None, feature=-1):
        """linear predictor
        compute the linear predictor portion of the model
        ie multiply the model matrix by the spline basis coefficients

        Parameters
        ---------
        at least 1 of (X, modelmat)
            and
        at least 1 of (b, feature)

        X : array-like of shape (n_samples, m_features), default: None
            containing the input dataset
            if None, will attempt to use modelmat

        modelmat : array-like, default: None
            contains the spline basis for each feature evaluated at the input
            values for each feature, ie model matrix
            if None, will attempt to construct the model matrix from X

        b : array-like, default: None
            contains the spline coefficients
            if None, will use current model coefficients

        feature : int, deafult: -1
                  feature for which to compute the linear prediction
                  if -1, will compute for all features

        Returns
        -------
        lp : np.array of shape (n_samples,)
        """
        if modelmat is None:
            modelmat = self._modelmat(X, feature=feature)
        if b is None:
            b = self.coef_[self._select_feature(feature)]
        return modelmat.dot(b).flatten()

    def predict_mu(self, X):
        """
        preduct expected value of target given model and input X

        Parameters
        ---------
        X : array-like of shape (n_samples, m_features), default: None
            containing the input dataset

        Returns
        -------
        y : np.array of shape (n_samples,)
            containing expected values under the model
        """
        if not self._is_fitted:
            raise AttributeError('GAM has not been fitted. Call fit first.')

        X = check_X(X, n_feats=len(self._n_coeffs) - self._fit_intercept,
                    edge_knots=self._edge_knots, dtypes=self._dtype)

        lp = self._linear_predictor(X)
        return self.link.mu(lp, self.distribution)

    def predict(self, X):
        """
        preduct expected value of target given model and input X
        often this is done via expected value of GAM given input X

        Parameters
        ---------
        X : array-like of shape (n_samples, m_features), default: None
            containing the input dataset

        Returns
        -------
        y : np.array of shape (n_samples,)
            containing predicted values under the model
        """
        if not self._is_fitted:
            raise AttributeError('GAM has not been fitted. Call fit first.')

        X = check_X(X, n_feats=len(self._n_coeffs) - self._fit_intercept,
                    edge_knots=self._edge_knots, dtypes=self._dtype)

        return self.predict_mu(X)

    def _modelmat(self, X, feature=-1):
        """
        Builds a model matrix, B, out of the spline basis for each feature

        B = [B_0, B_1, ..., B_p]

        Parameters
        ---------
        X : array-like of shape (n_samples, m_features), default: None
            containing the input dataset
        feature : int, default: -1
            feature index for which to compute the model matrix
            if -1, will create the model matrix for all features

        Returns
        -------
        modelmat : np.array of len n_samples
            containing model matrix of the spline basis for selected features
        """
        X = check_X(X, n_feats=len(self._n_coeffs) - self._fit_intercept,
                    edge_knots=self._edge_knots, dtypes=self._dtype)

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

    def _cholesky(self, A, **kwargs):
        """
        method to handle potential problems with the cholesky decomposition.

        will try to increase L2 regularization of the penalty matrix to
        do away with non-positive-definite errors

        Parameters
        ----------
        A : np.array

        Returns
        -------
        np.array
        """
        # create appropriate-size diagonal matrix
        if sp.sparse.issparse(A):
            diag = sp.sparse.eye(A.shape[0])
        else:
            diag = np.eye(A.shape[0])

        constraint_l2 = self._constraint_l2
        while constraint_l2 <= self._constraint_l2_max:
            try:
                L = cholesky(A, **kwargs)
                self._constraint_l2 = constraint_l2
                return L
            except NotPositiveDefiniteError:
                warnings.warn('Matrix is not positive definite. \n'\
                              'Increasing l2 reg by factor of 10.',
                              stacklevel=2)
                A -= constraint_l2 * diag
                constraint_l2 *= 10
                A += constraint_l2 * diag

        raise NotPositiveDefiniteError('Matrix is not positive \n'
                                       'definite.')


    def _C(self):
        """
        builds the GAM block-diagonal constraint matrix in quadratic form
        out of constraint matrices specified for each feature.

        behaves like a penalty, but with a very large lambda value, ie 1e6.

        Parameters
        ---------
        None

        Returns
        -------
        C : sparse CSC matrix containing the model constraints in quadratic form
        """
        Cs = []

        if self._fit_intercept:
            Cs.append(np.array(0.))

        for i, c in enumerate(self._constraints):
            fit_linear = self._fit_linear[i]
            dtype = self._dtype[i]
            n = self._n_coeffs[i + self._fit_intercept]
            coef = self.coef_[self._select_feature(i + self._fit_intercept)]
            coef = coef[fit_linear:]

            if c is None:
                c = 'none'
            if c in CONSTRAINTS:
                c = CONSTRAINTS[c]

            c = wrap_penalty(c, fit_linear)(n, coef) * self._constraint_lam
            Cs.append(c)

        Cs = sp.sparse.block_diag(Cs)

        # improve condition
        if Cs.nnz > 0:
            Cs += sp.sparse.diags(self._constraint_l2 * np.ones(Cs.shape[0]))

        return Cs

    def _P(self):
        """
        builds the GAM block-diagonal penalty matrix in quadratic form
        out of penalty matrices specified for each feature.

        each feature penalty matrix is multiplied by a lambda for that feature.
        the first feature is the intercept.

        so for m features:
        P = block_diag[lam0 * P0, lam1 * P1, lam2 * P2, ... , lamm * Pm]


        Parameters
        ---------
        None

        Returns
        -------
        P : sparse CSC matrix containing the model penalties in quadratic form

        """
        Ps = []

        if self._fit_intercept:
            Ps.append(np.array(0.))

        for i, p in enumerate(self._penalties):
            fit_linear = self._fit_linear[i]
            dtype = self._dtype[i]
            n = self._n_coeffs[i + self._fit_intercept]
            coef = self.coef_[self._select_feature(i + self._fit_intercept)]
            coef = coef[fit_linear:]

            if p == 'auto':
                if dtype == 'numerical':
                    p = derivative
                if dtype == 'categorical':
                    p = l2
            if p is None:
                p = 'none'
            if p in PENALTIES:
                p = PENALTIES[p]

            p = wrap_penalty(p, fit_linear)(n, coef)
            Ps.append(p)

        P_matrix = tuple([np.multiply(P, lam) for lam, P in zip(self._lam, Ps)])
        P_matrix = sp.sparse.block_diag(P_matrix)

        return P_matrix

    def _pseudo_data(self, y, lp, mu):
        """
        compute the pseudo data for a PIRLS iterations

        Parameters
        ---------
        y : array-like of shape (n,)
            containing target data
        lp : array-like of shape (n,)
            containing linear predictions by the model
        mu : array-like of shape (n_samples,)
            expected value of the targets given the model and inputs

        Returns
        -------
        pseudo_data : np.array of shape (n,)
        """
        return lp + (y - mu) * self.link.gradient(mu, self.distribution)

    def _W(self, mu, weights):
        """
        compute the PIRLS weights for model predictions.

        TODO lets verify the formula for this.
        if we use the square root of the mu with the stable opt,
        we get the same results as when we use non-sqrt mu with naive opt.

        this makes me think that they are equivalent.

        also, using non-sqrt mu with stable opt gives very small edofs for even lam=0.001
        and the parameter variance is huge. this seems strange to me.

        computed [V * d(link)/d(mu)] ^(-1/2) by hand and the math checks out as hoped.

        ive since moved the square to the naive pirls method to make the code modular.

        Parameters
        ---------
        mu : array-like of shape (n_samples,)
            expected value of the targets given the model and inputs
        weights : array-like of shape (n_samples,)
            containing sample weights

        Returns
        -------
        weights : sp..sparse array of shape (n_samples, n_samples)
        """
        return sp.sparse.diags((self.link.gradient(mu, self.distribution)**2 *
                                self.distribution.V(mu=mu) *
                                weights ** -1)**-0.5)

    def _mask(self, weights):
        """
        identifies the mask at which the weights are
            greater than sqrt(machine epsilon)
        and
            not NaN
        and
            not Inf


        Parameters
        ---------
        weights : array-like of shape (n,)
            containing weights in [0,1]

        Returns
        -------
        mask : boolean np.array of shape (n,) of good weight values
        """
        mask = (np.abs(weights) >= np.sqrt(EPS)) * np.isfinite(weights)
        assert mask.sum() != 0, 'increase regularization'
        return mask

    def _pirls(self, X, Y, weights):
        """
        Performs stable PIRLS iterations to estimate GAM coefficients

        Parameters
        ---------
        X : array-like of shape (n_samples, m_features)
            containing input data
        Y : array-like of shape (n,)
            containing target data
        weights : array-like of shape (n,)
            containing sample weights

        Returns
        -------
        None
        """
        modelmat = self._modelmat(X) # build a basis matrix for the GLM
        n, m = modelmat.shape
        min_n_m = np.min([m,n])

        # initialize GLM coefficients
        if not self._is_fitted or len(self.coef_) != sum(self._n_coeffs):
            self.coef_ = np.ones(m) * np.sqrt(EPS) # allow more training

        # do our penalties require recomputing cholesky?
        chol_pen = np.ravel([np.ravel(p) for p in self._penalties])
        chol_pen = any([cp in ['convex', 'concave', 'monotonic_inc',
                               'monotonic_dec', 'circular']for cp in chol_pen])
        P = self._P() # create penalty matrix

        # base penalty
        S = sp.sparse.diags(np.ones(m) * np.sqrt(EPS)) # improve condition
        # S += self._H # add any user-chosen minumum penalty to the diagonal

        # if we dont have any constraints, then do cholesky now
        if not any(self._constraints) and not chol_pen:
            E = self._cholesky(S + P, sparse=False)

        Dinv = np.zeros((min_n_m + m, m)).T

        for _ in range(self.max_iter):

            # recompute cholesky if needed
            if any(self._constraints) or chol_pen:
                P = self._P()
                C = self._C()
                E = self._cholesky(S + P + C, sparse=False)

            # forward pass
            y = deepcopy(Y) # for simplicity
            lp = self._linear_predictor(modelmat=modelmat)
            mu = self.link.mu(lp, self.distribution)
            W = self._W(mu, weights) # create pirls weight matrix

            # check for weghts == 0, nan, and update
            mask = self._mask(W.diagonal())
            y = y[mask] # update
            lp = lp[mask] # update
            mu = mu[mask] # update
            W = sp.sparse.diags(W.diagonal()[mask]) # update

            # PIRLS Wood pg 183
            pseudo_data = W.dot(self._pseudo_data(y, lp, mu))

            # log on-loop-start stats
            self._on_loop_start(vars())

            WB = W.dot(modelmat[mask,:]) # common matrix product
            Q, R = np.linalg.qr(WB.todense())

            if not np.isfinite(Q).all() or not np.isfinite(R).all():
                raise ValueError('QR decomposition produced NaN or Inf. '\
                                     'Check X data.')

            U, d, Vt = np.linalg.svd(np.vstack([R, E.T]))
            svd_mask = d <= (d.max() * np.sqrt(EPS)) # mask out small singular values

            np.fill_diagonal(Dinv, d**-1) # invert the singular values
            U1 = U[:min_n_m,:] # keep only top portion of U

            B = Vt.T.dot(Dinv).dot(U1.T).dot(Q.T)
            coef_new = B.dot(pseudo_data).A.flatten()
            diff = np.linalg.norm(self.coef_ - coef_new)/np.linalg.norm(coef_new)
            self.coef_ = coef_new # update

            # log on-loop-end stats
            self._on_loop_end(vars())

            # check convergence
            if diff < self.tol:
                break

        # estimate statistics even if not converged
        self._estimate_model_statistics(Y, modelmat, inner=None, BW=WB.T, B=B,
                                        weights=weights)
        if diff < self.tol:
            return

        print('did not converge')
        return

    def _pirls_naive(self, X, y):
        """
        Performs naive PIRLS iterations to estimate GAM coefficients

        Parameters
        ---------
        X : array-like of shape (n_samples, m_features)
            containing input data
        y : array-like of shape (n,)
            containing target data

        Returns
        -------
        None
        """
        modelmat = self._modelmat(X) # build a basis matrix for the GLM
        m = modelmat.shape[1]

        # initialize GLM coefficients
        if not self._is_fitted or len(self.coef_) != sum(self._n_coeffs):
            self.coef_ = np.ones(m) * np.sqrt(EPS) # allow more training

        P = self._P() # create penalty matrix
        P += sp.sparse.diags(np.ones(m) * np.sqrt(EPS)) # improve condition

        for _ in range(self.max_iter):
            lp = self._linear_predictor(modelmat=modelmat)
            mu = self.link.mu(lp, self.distribution)

            mask = self._mask(mu)
            mu = mu[mask] # update
            lp = lp[mask] # update

            if self.family == 'binomial':
                self.acc.append(self.accuracy(y=y[mask], mu=mu)) # log the training accuracy
            self.dev.append(self.deviance_(y=y[mask], mu=mu, scaled=False)) # log the training deviance

            weights = self._W(mu)**2 # PIRLS, added square for modularity
            pseudo_data = self._pseudo_data(y, lp, mu) # PIRLS

            BW = modelmat.T.dot(weights).tocsc() # common matrix product
            inner = sp.sparse.linalg.inv(BW.dot(modelmat) + P) # keep for edof

            coef_new = inner.dot(BW).dot(pseudo_data).flatten()
            diff = np.linalg.norm(self.coef_ - coef_new)/np.linalg.norm(coef_new)
            self.diffs.append(diff)
            self.coef_ = coef_new # update

            # check convergence
            if diff < self.tol:
                self.edof_ = self._estimate_edof(modelmat, inner, BW)
                self.aic_ = self._estimate_AIC(X, y, mu)
                self.aicc_ = self._estimate_AICc(X, y, mu)
                return

        print('did not converge')

    def _on_loop_start(self, variables):
        """
        performs on-loop-start actions like callbacks

        variables contains local namespace variables.

        Parameters
        ---------
        variables : dict of available variables

        Returns
        -------
        None
        """
        for callback in self.callbacks:
            if hasattr(callback, 'on_loop_start'):
                self.logs_[str(callback)].append(callback.on_loop_start(**variables))

    def _on_loop_end(self, variables):
        """
        performs on-loop-end actions like callbacks

        variables contains local namespace variables.

        Parameters
        ---------
        variables : dict of available variables

        Returns
        -------
        None
        """
        for callback in self.callbacks:
            if hasattr(callback, 'on_loop_end'):
                self.logs_[str(callback)].append(callback.on_loop_end(**variables))

    def fit(self, X, y, weights=None):
        """Fit the generalized additive model.

        Parameters
        ----------
        X : array-like, shape (n_samples, m_features)
            Training vectors, where n_samples is the number of samples
            and m_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values (integers in classification, real numbers in
            regression)
            For classification, labels must correspond to classes.
        weights : array-like shape (n_samples,) or None, default: None
            containing sample weights
            if None, defaults to array of ones
        Returns
        -------
        self : object
            Returns fitted GAM object
        """

        # validate parameters
        self._validate_params()

        # validate data
        y = check_y(y, self.link, self.distribution)
        X = check_X(X)
        check_X_y(X, y)

        if weights is not None:
            weights = np.array(weights).astype('f')
            check_lengths(y, weights)
        else:
            weights = np.ones_like(y).astype('f')

        # validate data-dependent parameters
        self._validate_data_dep_params(X)

        # set up logging
        if not hasattr(self, 'logs_'):
            self.logs_ = defaultdict(list)

        # optimize
        if self._opt == 0:
            self._pirls(X, y, weights)
        if self._opt == 1:
            self._pirls_naive(X, y)
        return self

    def deviance_residuals(self, X, y, weights=None, scaled=False):
        """
        method to compute the deviance residuals of the model

        these are analogous to the residuals of an OLS.

        Parameters
        ----------
        X : array-like
          input data array of shape (n_saples, m_features)
        y : array-like
          output data vector of shape (n_samples,)
        weights : array-like shape (n_samples,) or None, default: None
            containing sample weights
            if None, defaults to array of ones
        scaled : bool, default: False
          whether to scale the deviance by the (estimated) distribution scale

        Returns
        -------
        deviance_residuals : np.array
          with shape (n_samples,)
        """
        if not self._is_fitted:
            raise AttributeError('GAM has not been fitted. Call fit first.')

        y = check_y(y, self.link, self.distribution)
        X = check_X(X, n_feats=len(self._n_coeffs) - self._fit_intercept,
                    edge_knots=self._edge_knots, dtypes=self._dtype)
        check_X_y(X, y)

        if weights is not None:
            weights = np.array(weights).astype('f')
            check_lengths(y, weights)
        else:
            weights = np.ones_like(y).astype('f')

        mu = self.predict_mu(X)
        sign = np.sign(y-mu)
        return sign * self.distribution.deviance(y, mu,
                                                 weights=weights,
                                                 scaled=scaled) ** 0.5

    def _estimate_model_statistics(self, y, modelmat, inner=None, BW=None,
                                   B=None, weights=None):
        """
        method to compute all of the model statistics

        results are stored in the 'statistics_' attribute of the model, as a
        dictionary keyed by:

        - edof: estimated degrees freedom
        - scale: distribution scale, if applicable
        - cov: coefficient covariances
        - AIC: Akaike Information Criterion
        - AICc: corrected Akaike Information Criterion
        - r2: explained_deviance Pseudo R-squared
        - GCV: generailized cross-validation
            or
        - UBRE: Un-Biased Risk Estimator

        Parameters
        ----------
        y : array-like
          output data vector of shape (n_samples,)
        modelmat : array-like, default: None
            contains the spline basis for each feature evaluated at the input
        inner : array of intermediate computations from naive optimization
        BW : array of intermediate computations from either optimization
        B : array of intermediate computations from stable optimization
        weights : array-like shape (n_samples,) or None, default: None
            containing sample weights

        Returns
        -------
        None
        """
        self.statistics_ = {}

        lp = self._linear_predictor(modelmat=modelmat)
        mu = self.link.mu(lp, self.distribution)
        self.statistics_['edof'] = self._estimate_edof(BW=BW, B=B)
        # self.edof_ = np.dot(U1, U1.T).trace().A.flatten() # this is wrong?
        if not self.distribution._known_scale:
            self.distribution.scale = self.distribution.phi(y=y, mu=mu, edof=self.statistics_['edof'], weights=weights)
        self.statistics_['scale'] = self.distribution.scale
        self.statistics_['cov'] = (B.dot(B.T)).A * self.distribution.scale # parameter covariances. no need to remove a W because we are using W^2. Wood pg 184
        self.statistics_['se'] = self.statistics_['cov'].diagonal()**0.5
        self.statistics_['AIC']= self._estimate_AIC(y=y, mu=mu, weights=weights)
        self.statistics_['AICc'] = self._estimate_AICc(y=y, mu=mu, weights=weights)
        self.statistics_['pseudo_r2'] = self._estimate_r2(y=y, mu=mu, weights=weights)
        self.statistics_['GCV'], self.statistics_['UBRE'] = self._estimate_GCV_UBRE(modelmat=modelmat, y=y, weights=weights)
        self.statistics_['loglikelihood'] = self._loglikelihood(y, mu, weights=weights)
        self.statistics_['deviance'] = self.distribution.deviance(y=y, mu=mu, weights=weights).sum()

    def _estimate_edof(self, modelmat=None, inner=None, BW=None, B=None,
                       limit=50000):
        """
        estimate effective degrees of freedom.

        computes the only diagonal of the influence matrix and sums.
        allows for subsampling when the number of samples is very large.

        Parameters
        ----------
        modelmat : array-like, default: None
            contains the spline basis for each feature evaluated at the input
        inner : array of intermediate computations from naive optimization
        BW : array of intermediate computations from either optimization
        B : array of intermediate computations from stable optimization
        limit : int, default: 50000
            number of samples required before subsampling the model matrix.
            this requires less computation.

        Returns
        -------
        None
        """
        size = BW.shape[1] # number of samples
        max_ = np.min([limit, size]) # since we only compute the diagonal, we can afford larger matrices
        if max_ == limit:
            # subsampling
            scale = np.float(size)/max_
            idxs = list(range(size))
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

    def _estimate_AIC(self, y, mu, weights=None):
        """
        estimate the Akaike Information Criterion

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            output data vector
        mu : array-like of shape (n_samples,)
            expected value of the targets given the model and inputs

        Returns
        -------
        None
        """
        estimated_scale = not(self.distribution._known_scale) # if we estimate the scale, that adds 2 dof
        return -2*self._loglikelihood(y=y, mu=mu, weights=weights) + \
                2*self.statistics_['edof'] + 2*estimated_scale

    def _estimate_AICc(self, y, mu, weights=None):
        """
        estimate the corrected Akaike Information Criterion

        relies on the estimated degrees of freedom, which must be computed
        before.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            output data vector
        mu : array-like of shape (n_samples,)
            expected value of the targets given the model and inputs

        Returns
        -------
        None
        """
        edof = self.statistics_['edof']
        if self.statistics_['AIC'] is None:
            self.statistics_['AIC'] = self._estimate_AIC(y, mu, weights)
        return self.statistics_['AIC'] + 2*(edof + 1)*(edof + 2)/(y.shape[0] - edof -2)

    def _estimate_r2(self, X=None, y=None, mu=None, weights=None):
        """
        estimate some pseudo R^2 values

        currently only computes explained deviance.
        results are stored

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            output data vector
        mu : array-like of shape (n_samples,)
            expected value of the targets given the model and inputs
        weights : array-like shape (n_samples,) or None, default: None
            containing sample weights
            if None, defaults to array of ones

        Returns
        -------
        None
        """
        if mu is None:
            mu = self.predict_mu_(X=X)

        if weights is None:
            weights = np.ones_like(y)

        null_mu = y.mean() * np.ones_like(y)

        null_d = self.distribution.deviance(y=y, mu=null_mu, weights=weights)
        full_d = self.distribution.deviance(y=y, mu=mu, weights=weights)

        null_ll = self._loglikelihood(y=y, mu=null_mu, weights=weights)
        full_ll = self._loglikelihood(y=y, mu=mu, weights=weights)

        r2 = OrderedDict()
        r2['explained_deviance'] = 1. - full_d.sum()/null_d.sum()
        r2['McFadden'] = 1. - full_ll/null_ll
        r2['McFadden_adj'] = 1. - (full_ll - self.statistics_['edof'])/null_ll

        null_ll = self._loglikelihood(y, mu, weights)
        return r2

    def _estimate_GCV_UBRE(self, X=None, y=None, modelmat=None, gamma=1.4,
                           add_scale=True, weights=None):
        """
        Generalized Cross Validation and Un-Biased Risk Estimator.

        UBRE is used when the scale parameter is known,
        like Poisson and Binomial families.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            output data vector
        modelmat : array-like, default: None
            contains the spline basis for each feature evaluated at the input
        gamma : float, default: 1.4
            serves as a weighting to increase the impact of the influence matrix
            on the score
        add_scale : boolean, default: True
            UBRE score can be negative because the distribution scale
            is subtracted. to keep things positive we can add the scale back.
        weights : array-like shape (n_samples,) or None, default: None
            containing sample weights
            if None, defaults to array of ones

        Returns
        -------
        score : float
            Either GCV or UBRE, depending on if the scale parameter is known.

        Notes
        -----
        Sometimes the GCV or UBRE selected model is deemed to be too wiggly,
        and a smoother model is desired. One way to achieve this, in a
        systematic way, is to increase the amount that each model effective
        degree of freedom counts, in the GCV or UBRE score, by a factor γ ≥ 1

        see Wood 2006 pg. 177-182, 220 for more details.
        """
        if gamma < 1:
            raise ValueError('gamma scaling should be greater than 1, '\
                             'but found gamma = {}',format(gamma))

        if modelmat is None:
            modelmat = self._modelmat(X)

        if weights is None:
            weights = np.ones_like(y)

        lp = self._linear_predictor(modelmat=modelmat)
        mu = self.link.mu(lp, self.distribution)
        n = y.shape[0]
        edof = self.statistics_['edof']

        GCV = None
        UBRE = None

        dev = self.distribution.deviance(mu=mu, y=y, scaled=False, weights=weights).sum()

        if self.distribution._known_scale:
            # scale is known, use UBRE
            scale = self.distribution.scale
            UBRE = 1./n *  dev - (~add_scale)*(scale) + 2.*gamma/n * edof * scale
        else:
            # scale unkown, use GCV
            GCV = (n * dev) / (n - gamma * edof)**2
        return (GCV, UBRE)

    def confidence_intervals(self, X, width=.95, quantiles=None):
        """
        estimate confidence intervals for the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, m_features)
            input data matrix
        width : float on [0,1], default: 0.95
        quantiles : array-like of floats in [0, 1], default: None
            instead of specifying the prediciton width, one can specify the
            quantiles. so width=.95 is equivalent to quantiles=[.025, .975]

        Returns
        -------
        intervals: np.array of shape (n_samples, 2 or len(quantiles))
        """
        if not self._is_fitted:
            raise AttributeError('GAM has not been fitted. Call fit first.')

        X = check_X(X, n_feats=len(self._n_coeffs) - self._fit_intercept,
                    edge_knots=self._edge_knots, dtypes=self._dtype)

        return self._get_quantiles(X, width, quantiles, prediction=False)

    def _get_quantiles(self, X, width, quantiles, modelmat=None, lp=None,
                       prediction=False, xform=True, feature=-1):
        """
        estimate prediction intervals for LinearGAM

        Parameters
        ----------
        X : array
            input data of shape (n_samples, m_features)
        y : array
            label data of shape (n_samples,)
        width : float on [0,1]
        quantiles : array-like of floats in [0, 1]
            instead of specifying the prediciton width, one can specify the
            quantiles. so width=.95 is equivalent to quantiles=[.025, .975]
        modelmat : array of shape

        Returns
        -------
        intervals: np.array of shape (n_samples, 2 or len(quantiles))
        """
        if quantiles is not None:
            quantiles = np.atleast_1d(quantiles)
        else:
            alpha = (1 - width)/2.
            quantiles = [alpha, 1 - alpha]
        for quantile in quantiles:
            if (quantile > 1) or (quantile < 0):
                raise ValueError('quantiles must be in [0, 1], but found {}'\
                                 .format(quantiles))

        if modelmat is None:
            modelmat = self._modelmat(X, feature=feature)
        if lp is None:
            lp = self._linear_predictor(modelmat=modelmat, feature=feature)

        idxs = self._select_feature(feature)
        cov = self.statistics_['cov'][idxs][:,idxs]

        var = (modelmat.dot(cov) * modelmat.todense().A).sum(axis=1)
        if prediction:
            var += self.distribution.scale

        lines = []
        for quantile in quantiles:
            t = sp.stats.t.ppf(quantile, df=self.statistics_['edof'])
            lines.append(lp + t * var**0.5)
        lines = np.vstack(lines).T

        if xform:
            lines = self.link.mu(lines, self.distribution)
        return lines

    def _select_feature(self, feature):
        """
        tool for indexing by feature function.

        many coefficients and parameters are organized by feature.
        this tool returns all of the indices for a given feature.

        GAM intercept is considered the 0th feature.

        Parameters
        ----------
        feature : int
            feature to select from the data.
            when fit_intercept=True, 0 corresponds to the intercept
            when feature=-1, all features are selected

        Returns
        -------
        np.array
            indices into self.coef_ corresponding to the chosen feature
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

    def partial_dependence(self, X, feature=-1, width=None, quantiles=None):
        """
        Computes the feature functions for the GAM
        and possibly their confidence intervals.

        if both width=None and quantiles=None,
        then no confidence intervals are computed

        Parameters
        ----------
        X : array
            input data of shape (n_samples, m_features)
        feature : array-like of ints, default: -1
            feature for which to compute the partial dependence functions
            if feature == -1, then all features are selected,
            excluding the intercept
            if feature == 0 and gam.fit_intercept is True, then the intercept's
            patial dependence is returned
        width : float in [0, 1], default: None
            width of the confidence interval
            if None, defaults to 0.95
        quantiles : array-like of floats in [0, 1], default: None
            instead of specifying the prediciton width, one can specify the
            quantiles. so width=.95 is equivalent to quantiles=[.025, .975]
            if None, defaults to width

        Returns
        -------
        pdeps : np.array of shape (n_samples, len(feature))
        conf_intervals : list of length len(feature)
            containing np.arrays of shape (n_samples, 2 or len(quantiles))
        """
        if not self._is_fitted:
            raise AttributeError('GAM has not been fitted. Call fit first.')

        m = len(self._n_coeffs) - self._fit_intercept
        X = check_X(X, n_feats=m, edge_knots=self._edge_knots,
                    dtypes=self._dtype)
        p_deps = []

        compute_quantiles = (width is not None) or (quantiles is not None)
        conf_intervals = []

        if feature == -1:
            feature = np.arange(m) + self._fit_intercept

        # convert to array
        feature = np.atleast_1d(feature)

        # ensure feature exists
        if (feature >= len(self._n_coeffs)).any() or (feature < -1).any():
            raise ValueError('feature {} out of range for X with shape {}'\
                             .format(feature, X.shape))

        for i in feature:
            modelmat = self._modelmat(X, feature=i)
            lp = self._linear_predictor(modelmat=modelmat, feature=i)
            p_deps.append(lp)

            if compute_quantiles:
                conf_intervals.append(self._get_quantiles(X, width=width,
                                                          quantiles=quantiles,
                                                          modelmat=modelmat,
                                                          lp=lp,
                                                          feature=i,
                                                          xform=False))
        pdeps = np.vstack(p_deps).T
        if compute_quantiles:
            return (pdeps, conf_intervals)
        return pdeps

    def summary(self):
        """
        produce a summary of the model statistics

        #TODO including feature significance via F-Test

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if not self._is_fitted:
            raise AttributeError('GAM has not been fitted. Call fit first.')

        keys = ['edof', 'AIC', 'AICc']
        if self.distribution._known_scale:
            keys.append('UBRE')
        else:
            keys.append('GCV')
        keys.append('loglikelihood')
        keys.append('deviance')
        keys.append('scale')

        sub_data = OrderedDict([[k, self.statistics_[k]] for k in keys])

        print_data(sub_data, title='Model Statistics')
        print('')
        print_data(self.statistics_['pseudo_r2'], title='Pseudo-R^2')

    def gridsearch(self, X, y, weights=None, return_scores=False,
                   keep_best=True, objective='auto', **param_grids):
        """
        performs a grid search over a space of parameters for a given objective

        NOTE:
        gridsearch method is lazy and will not remove useless combinations
        from the search space, eg.
          n_splines=np.arange(5,10), fit_splines=[True, False]
        will result in 10 loops, of which 5 are equivalent because
        even though fit_splines==False

        it is not recommended to search over a grid that alternates
        between known scales and unknown scales, as the scores of the
        cadidate models will not be comparable.

        Parameters
        ----------
        X : array
          input data of shape (n_samples, m_features)

        y : array
          label data of shape (n_samples,)

        weights : array-like shape (n_samples,) or None, default: None
            containing sample weights
            if None, defaults to array of ones

        return_scores : boolean, default False
          whether to return the hyperpamaters
          and score for each element in the grid

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
        if return_scores == True:
            model_scores : dict
                Contains each fitted model as keys and corresponding
                objective scores as values
        else:
            self, ie possibly the newly fitted model
        """
        # validate objective
        if objective not in ['auto', 'GCV', 'UBRE', 'AIC', 'AICc']:
            raise ValueError("objective mut be in "\
                             "['auto', 'GCV', 'UBRE', 'AIC', 'AICc'], '\
                             'but found objective = {}".format(objective))

        # check if model fitted
        if not self._is_fitted:
            self._validate_params()

        # check objective
        if self.distribution._known_scale:
            if objective == 'GCV':
                raise ValueError('GCV should be used for models with'\
                                 'unknown scale')
            if objective == 'auto':
                objective = 'UBRE'

        else:
            if objective == 'UBRE':
                raise ValueError('UBRE should be used for models with '\
                                 'known scale')
            if objective == 'auto':
                objective = 'GCV'

        # if no params, then set up default gridsearch
        if not bool(param_grids):
            param_grids['lam'] = np.logspace(-3, 3, 11)

        # validate params
        admissible_params = self.get_params()
        params = []
        grids = []
        for param, grid in list(param_grids.items()):

            if param not in (admissible_params):
                raise ValueError('unknown parameter: {}'.format(param))

            if not (isiterable(grid) and (len(grid) > 1)): \
                raise ValueError('{} grid must either be iterable of '
                                 'iterables, or an iterable of lengnth > 1, '\
                                 'but found {}'.format(param, grid))

            # prepare grid
            if any(isiterable(g) for g in grid):
                # cast to np.array
                grid = [np.atleast_1d(g) for g in grid]

                # set grid to combination of all grids
                grid = combine(*grid)

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

        # check if our model has been fitted already and store it
        if self._is_fitted:
            models.append(self)
            scores.append(self.statistics_[objective])

            # our model is currently the best
            best_model = models[-1]
            best_score = scores[-1]

        # loop through candidate model params
        pbar = ProgressBar()
        for param_grid in pbar(param_grid_list):

            # define new model
            gam = deepcopy(self)
            gam.set_params(self.get_params())
            gam.set_params(**param_grid)

            # warm start with parameters from previous build
            if models:
                coef = models[-1].coef_
                gam.set_params(coef_=coef, force=True)

            try:
                # try fitting
                gam.fit(X, y, weights)

            except ValueError as error:
                msg = str(error) + '\non model:\n' + str(gam)
                msg += '\nskipping...\n'
                warnings.warn(msg)
                continue

            # record results
            models.append(gam)
            scores.append(gam.statistics_[objective])

            # track best
            if scores[-1] < best_score:
                best_model = models[-1]
                best_score = scores[-1]

        # problems
        if len(models) == 0:
            msg = 'No models were fitted.'
            warnings.warn(msg)
            return self

        # copy over the best
        if keep_best:
            self.set_params(deep=True,
                            force=True,
                            **best_model.get_params(deep=True))
        if return_scores:
            return OrderedDict(zip(models, scores))
        else:
            return self


class LinearGAM(GAM):
    """Linear GAM

    Parameters
    ----------
    callbacks : list of strings or list of CallBack objects,
                default: ['deviance', 'diffs']
        Names of callback objects to call during the optimization loop.

    constraints : str or callable, or iterable of str or callable,
                  default: None
        Names of constraint functions to call during the optimization loop.

        Must be in {'convex', 'concave', 'monotonic_inc', 'monotonic_dec',
                    'circular', 'none'}

        If None, then the model will apply no constraints.

        If only one str or callable is specified, then is it copied for all
        features.

    dtype : str in {'auto', 'numerical',  'categorical'},
            or list of str, default: 'auto'
        String describing the data-type of each feature.

        'numerical' is used for continuous-valued data-types,
            like in regression.
        'categorical' is used for discrete-valued data-types,
            like in classification.

        If only one str is specified, then is is copied for all features.

    lam : float or iterable of floats > 0, default: 0.6
        Smoothing strength; must be a positive float, or one positive float
        per feature.

        Larger values enforce stronger smoothing.

        If only one float is specified, then it is copied for all features.

    fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

        NOTE: the intercept receives no smoothing penalty.

    fit_linear : bool or iterable of bools, default: False
        Specifies if a linear term should be added to any of the feature
        functions. Useful for including pre-defined feature transformations
        in the model.

        If only one bool is specified, then it is copied for all features.

        NOTE: Many constraints are incompatible with an additional linear fit.
            eg. if a non-zero linear function is added to a periodic spline
            function, it will cease to be periodic.

            this is also possible for a monotonic spline function.

    fit_splines : bool or iterable of bools, default: True
        Specifies if a smoother should be added to any of the feature
        functions. Useful for defining feature transformations a-priori
        that should not have splines fitted to them.

        If only one bool is specified, then it is copied for all features.

        NOTE: fit_splines supercedes n_splines.
        ie. if n_splines > 0 and fit_splines = False, no splines will be fitted.

    max_iter : int, default: 100
        Maximum number of iterations allowed for the solver to converge.

    penalties : str or callable, or iterable of str or callable,
                default: 'auto'
        Type of penalty to use for each feature.

        penalty should be in {'auto', 'none', 'derivative', 'l2', }

        If 'auto', then the model will use 2nd derivative smoothing for features
        of dtype 'numerical', and L2 smoothing for features of dtype
        'categorical'.

        If only one str or callable is specified, then is it copied for all
        features.

    n_splines : int, or iterable of ints, default: 25
        Number of splines to use in each feature function; must be non-negative.
        If only one int is specified, then it is copied for all features.

        Note: this value is set to 0 if fit_splines is False

    scale : float or None, default: None
        scale of the distribution, if known a-priori.
        if None, scale is estimated.

    spline_order : int, or iterable of ints, default: 3
        Order of spline to use in each feature function; must be non-negative.
        If only one int is specified, then it is copied for all features

        Note: if a feature is of type categorical, spline_order will be set to 0.

    tol : float, default: 1e-4
        Tolerance for stopping criteria.

    Attributes
    ----------
    coef_ : array, shape (n_classes, m_features)
        Coefficient of the features in the decision function.
        If fit_intercept is True, then self.coef_[0] will contain the bias.

    statistics_ : dict
        Dictionary containing model statistics like GCV/UBRE scores, AIC/c,
        parameter covariances, estimated degrees of freedom, etc.

    logs_ : dict
        Dictionary containing the outputs of any callbacks at each
        optimization loop.

        The logs are structured as `{callback: [...]}`

    References
    ----------
    Simon N. Wood, 2006
    Generalized Additive Models: an introduction with R

    Hastie, Tibshirani, Friedman
    The Elements of Statistical Learning
    http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf

    Paul Eilers & Brian Marx, 2015
    International Biometric Society: A Crash Course on P-splines
    http://www.ibschannel2015.nl/project/userfiles/Crash_course_handout.pdf
    """
    def __init__(self, lam=0.6, max_iter=100, n_splines=25, spline_order=3,
                 penalties='auto', dtype='auto', tol=1e-4, scale=None,
                 callbacks=['deviance', 'diffs'],
                 fit_intercept=True, fit_linear=False, fit_splines=True,
                 constraints=None):
        self.scale = scale
        super(LinearGAM, self).__init__(distribution=NormalDist(scale=self.scale),
                                        link='identity',
                                        lam=lam,
                                        dtype=dtype,
                                        max_iter=max_iter,
                                        n_splines=n_splines,
                                        spline_order=spline_order,
                                        penalties=penalties,
                                        tol=tol,
                                        callbacks=callbacks,
                                        fit_intercept=fit_intercept,
                                        fit_linear=fit_linear,
                                        fit_splines=fit_splines,
                                        constraints=constraints)

        self._exclude += ['distribution', 'link']

    def _validate_params(self):
        """
        method to sanitize model parameters

        Parameters
        ---------
        None

        Returns
        -------
        None
        """
        self.distribution = NormalDist(scale=self.scale)
        super(LinearGAM, self)._validate_params()

    def prediction_intervals(self, X, width=.95, quantiles=None):
        """
        estimate prediction intervals for LinearGAM

        Parameters
        ----------
        X : array-like of shape (n_samples, m_features)
            input data matrix
        width : float on [0,1], default: 0.95
        quantiles : array-like of floats in [0, 1], default: None
            instead of specifying the prediciton width, one can specify the
            quantiles. so width=.95 is equivalent to quantiles=[.025, .975]

        Returns
        -------
        intervals: np.array of shape (n_samples, 2 or len(quantiles))
        """
        if not self._is_fitted:
            raise AttributeError('GAM has not been fitted. Call fit first.')

        X = check_X(X, n_feats=len(self._n_coeffs) - self._fit_intercept,
                    edge_knots=self._edge_knots, dtypes=self._dtype)

        return self._get_quantiles(X, width, quantiles, prediction=True)

class LogisticGAM(GAM):
    """Logistic GAM

    Parameters
    ----------
    callbacks : list of strings or list of CallBack objects,
                default: ['deviance', 'diffs']
        Names of callback objects to call during the optimization loop.

    constraints : str or callable, or iterable of str or callable,
                  default: None
        Names of constraint functions to call during the optimization loop.

        Must be in {'convex', 'concave', 'monotonic_inc', 'monotonic_dec',
                    'circular', 'none'}

        If None, then the model will apply no constraints.

        If only one str or callable is specified, then is it copied for all
        features.

    dtype : str in {'auto', 'numerical',  'categorical'},
            or list of str, default: 'auto'
        String describing the data-type of each feature.

        'numerical' is used for continuous-valued data-types,
            like in regression.
        'categorical' is used for discrete-valued data-types,
            like in classification.

        If only one str is specified, then is is copied for all features.

    lam : float or iterable of floats > 0, default: 0.6
        Smoothing strength; must be a positive float, or one positive float
        per feature.

        Larger values enforce stronger smoothing.

        If only one float is specified, then it is copied for all features.

    fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

        NOTE: the intercept receives no smoothing penalty.

    fit_linear : bool or iterable of bools, default: False
        Specifies if a linear term should be added to any of the feature
        functions. Useful for including pre-defined feature transformations
        in the model.

        If only one bool is specified, then it is copied for all features.

        NOTE: Many constraints are incompatible with an additional linear fit.
            eg. if a non-zero linear function is added to a periodic spline
            function, it will cease to be periodic.

            this is also possible for a monotonic spline function.

    fit_splines : bool or iterable of bools, default: True
        Specifies if a smoother should be added to any of the feature
        functions. Useful for defining feature transformations a-priori
        that should not have splines fitted to them.

        If only one bool is specified, then it is copied for all features.

        NOTE: fit_splines supercedes n_splines.
        ie. if n_splines > 0 and fit_splines = False, no splines will be fitted.

    max_iter : int, default: 100
        Maximum number of iterations allowed for the solver to converge.

    penalties : str or callable, or iterable of str or callable,
                default: 'auto'
        Type of penalty to use for each feature.

        penalty should be in {'auto', 'none', 'derivative', 'l2', }

        If 'auto', then the model will use 2nd derivative smoothing for features
        of dtype 'numerical', and L2 smoothing for features of dtype
        'categorical'.

        If only one str or callable is specified, then is it copied for all
        features.

    n_splines : int, or iterable of ints, default: 25
        Number of splines to use in each feature function; must be non-negative.
        If only one int is specified, then it is copied for all features.

        Note: this value is set to 0 if fit_splines is False

    spline_order : int, or iterable of ints, default: 3
        Order of spline to use in each feature function; must be non-negative.
        If only one int is specified, then it is copied for all features

        Note: if a feature is of type categorical, spline_order will be set to 0.

    tol : float, default: 1e-4
        Tolerance for stopping criteria.

    Attributes
    ----------
    coef_ : array, shape (n_classes, m_features)
        Coefficient of the features in the decision function.
        If fit_intercept is True, then self.coef_[0] will contain the bias.

    statistics_ : dict
        Dictionary containing model statistics like GCV/UBRE scores, AIC/c,
        parameter covariances, estimated degrees of freedom, etc.

    logs_ : dict
        Dictionary containing the outputs of any callbacks at each
        optimization loop.

        The logs are structured as `{callback: [...]}`

    References
    ----------
    Simon N. Wood, 2006
    Generalized Additive Models: an introduction with R

    Hastie, Tibshirani, Friedman
    The Elements of Statistical Learning
    http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf

    Paul Eilers & Brian Marx, 2015
    International Biometric Society: A Crash Course on P-splines
    http://www.ibschannel2015.nl/project/userfiles/Crash_course_handout.pdf
    """
    def __init__(self, lam=0.6, max_iter=100, n_splines=25, spline_order=3,
                 penalties='auto', dtype='auto', tol=1e-4,
                 callbacks=['deviance', 'diffs', 'accuracy'],
                 fit_intercept=True, fit_linear=False, fit_splines=True,
                 constraints=None):

        # call super
        super(LogisticGAM, self).__init__(distribution='binomial',
                                          link='logit',
                                          lam=lam,
                                          dtype=dtype,
                                          max_iter=max_iter,
                                          n_splines=n_splines,
                                          spline_order=spline_order,
                                          penalties=penalties,
                                          tol=tol,
                                          callbacks=callbacks,
                                          fit_intercept=fit_intercept,
                                          fit_linear=fit_linear,
                                          fit_splines=fit_splines,
                                          constraints=constraints)
        # ignore any variables
        self._exclude += ['distribution', 'link']

    def accuracy(self, X=None, y=None, mu=None):
        """
        computes the accuracy of the LogisticGAM

        Parameters
        ----------
        note: X or mu must be defined. defaults to mu

        X : array-like of shape (n_samples, m_features), default: None
            containing input data
        y : array-like of shape (n,)
            containing target data
        mu : array-like of shape (n_samples,), default: None
            expected value of the targets given the model and inputs

        Returns
        -------
        float in [0, 1]
        """
        if not self._is_fitted:
            raise AttributeError('GAM has not been fitted. Call fit first.')

        y = check_y(y, self.link, self.distribution)
        if X is not None:
            X = check_X(X, n_feats=len(self._n_coeffs) - self._fit_intercept,
                        edge_knots=self._edge_knots, dtypes=self._dtype)

        if mu is None:
            mu = self.predict_mu(X)
        check_X_y(mu, y)
        return ((mu > 0.5).astype(int) == y).mean()

    def predict(self, X):
        """
        preduct binary targets given model and input X

        Parameters
        ---------
        X : array-like of shape (n_samples, m_features), default: None
            containing the input dataset

        Returns
        -------
        y : np.array of shape (n_samples,)
            containing binary targets under the model
        """
        return self.predict_mu(X) > 0.5

    def predict_proba(self, X):
        """
        preduct targets given model and input X

        Parameters
        ---------
        X : array-like of shape (n_samples, m_features), default: None
            containing the input dataset

        Returns
        -------
        y : np.array of shape (n_samples,)
            containing expected values under the model
        """
        return self.predict_mu(X)


class PoissonGAM(GAM):
    """Poisson GAM

    Parameters
    ----------
    callbacks : list of strings or list of CallBack objects,
                default: ['deviance', 'diffs']
        Names of callback objects to call during the optimization loop.

    constraints : str or callable, or iterable of str or callable,
                  default: None
        Names of constraint functions to call during the optimization loop.

        Must be in {'convex', 'concave', 'monotonic_inc', 'monotonic_dec',
                    'circular', 'none'}

        If None, then the model will apply no constraints.

        If only one str or callable is specified, then is it copied for all
        features.

    dtype : str in {'auto', 'numerical',  'categorical'},
            or list of str, default: 'auto'
        String describing the data-type of each feature.

        'numerical' is used for continuous-valued data-types,
            like in regression.
        'categorical' is used for discrete-valued data-types,
            like in classification.

        If only one str is specified, then is is copied for all features.

    lam : float or iterable of floats > 0, default: 0.6
        Smoothing strength; must be a positive float, or one positive float
        per feature.

        Larger values enforce stronger smoothing.

        If only one float is specified, then it is copied for all features.

    fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

        NOTE: the intercept receives no smoothing penalty.

    fit_linear : bool or iterable of bools, default: False
        Specifies if a linear term should be added to any of the feature
        functions. Useful for including pre-defined feature transformations
        in the model.

        If only one bool is specified, then it is copied for all features.

        NOTE: Many constraints are incompatible with an additional linear fit.
            eg. if a non-zero linear function is added to a periodic spline
            function, it will cease to be periodic.

            this is also possible for a monotonic spline function.

    fit_splines : bool or iterable of bools, default: True
        Specifies if a smoother should be added to any of the feature
        functions. Useful for defining feature transformations a-priori
        that should not have splines fitted to them.

        If only one bool is specified, then it is copied for all features.

        NOTE: fit_splines supercedes n_splines.
        ie. if n_splines > 0 and fit_splines = False, no splines will be fitted.

    max_iter : int, default: 100
        Maximum number of iterations allowed for the solver to converge.

    penalties : str or callable, or iterable of str or callable,
                default: 'auto'
        Type of penalty to use for each feature.

        penalty should be in {'auto', 'none', 'derivative', 'l2', }

        If 'auto', then the model will use 2nd derivative smoothing for features
        of dtype 'numerical', and L2 smoothing for features of dtype
        'categorical'.

        If only one str or callable is specified, then is it copied for all
        features.

    n_splines : int, or iterable of ints, default: 25
        Number of splines to use in each feature function; must be non-negative.
        If only one int is specified, then it is copied for all features.

        Note: this value is set to 0 if fit_splines is False

    spline_order : int, or iterable of ints, default: 3
        Order of spline to use in each feature function; must be non-negative.
        If only one int is specified, then it is copied for all features

        Note: if a feature is of type categorical, spline_order will be set to 0.

    tol : float, default: 1e-4
        Tolerance for stopping criteria.

    Attributes
    ----------
    coef_ : array, shape (n_classes, m_features)
        Coefficient of the features in the decision function.
        If fit_intercept is True, then self.coef_[0] will contain the bias.

    statistics_ : dict
        Dictionary containing model statistics like GCV/UBRE scores, AIC/c,
        parameter covariances, estimated degrees of freedom, etc.

    logs_ : dict
        Dictionary containing the outputs of any callbacks at each
        optimization loop.

        The logs are structured as `{callback: [...]}`

    References
    ----------
    Simon N. Wood, 2006
    Generalized Additive Models: an introduction with R

    Hastie, Tibshirani, Friedman
    The Elements of Statistical Learning
    http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf

    Paul Eilers & Brian Marx, 2015
    International Biometric Society: A Crash Course on P-splines
    http://www.ibschannel2015.nl/project/userfiles/Crash_course_handout.pdf
    """
    def __init__(self, lam=0.6, max_iter=100, n_splines=25, spline_order=3,
                 penalties='auto', dtype='auto', tol=1e-4,
                 callbacks=['deviance', 'diffs', 'accuracy'],
                 fit_intercept=True, fit_linear=False, fit_splines=True,
                 constraints=None):

        # call super
        super(PoissonGAM, self).__init__(distribution='poisson',
                                         link='log',
                                         lam=lam,
                                         dtype=dtype,
                                         max_iter=max_iter,
                                         n_splines=n_splines,
                                         spline_order=spline_order,
                                         penalties=penalties,
                                         tol=tol,
                                         callbacks=callbacks,
                                         fit_intercept=fit_intercept,
                                         fit_linear=fit_linear,
                                         fit_splines=fit_splines,
                                         constraints=constraints)
        # ignore any variables
        self._exclude += ['distribution', 'link']

    def _loglikelihood(self, y, mu, weights=None, rescale_y=True):
        """
        compute the log-likelihood of the dataset using the current model

        Parameters
        ---------
        y : array-like of shape (n,)
            containing target values
        mu : array-like of shape (n_samples,)
            expected value of the targets given the model and inputs
        weights : array-like of shape (n,)
            containing sample weights
        rescale_y : boolean, defaul: True
            whether to scale the targets back up by

        Returns
        -------
        log-likelihood : np.array of shape (n,)
            containing log-likelihood scores
        """
        if weights is not None:
            weights = np.array(weights).astype('f')
        else:
            weights = np.ones_like(y).astype('f')

        if rescale_y:
            y = y * weights

        return np.log(self.distribution.pdf(y=y, mu=mu, weights=weights)).sum()

    def loglikelihood(self, X, y, exposure=None, weights=None):
        """
        compute the log-likelihood of the dataset using the current model

        Parameters
        ---------
        X : array-like of shape (n_samples, m_features)
            containing the input dataset
        y : array-like of shape (n,)
            containing target values
        exposure : array-like shape (n_samples,) or None, default: None
            containing exposures
            if None, defaults to array of ones
        weights : array-like of shape (n,)
            containing sample weights

        Returns
        -------
        log-likelihood : np.array of shape (n,)
            containing log-likelihood scores
        """
        mu = self.predict_mu(X)
        y, weights = self._exposure_to_weights(y, exposure, weights)
        return self._loglikelihood(y, mu, weights=weights, rescale_y=True)

    def _exposure_to_weights(self, y, exposure=None, weights=None):
        """simple tool to create a common API

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target values (integers in classification, real numbers in
            regression)
            For classification, labels must correspond to classes.
        exposure : array-like shape (n_samples,) or None, default: None
            containing exposures
            if None, defaults to array of ones
        weights : array-like shape (n_samples,) or None, default: None
            containing sample weights
            if None, defaults to array of ones

        Returns
        -------
        y : y normalized by exposure
        weights : array-like shape (n_samples,)
        """

        if exposure is not None:
            exposure = np.array(exposure).astype('f')
        else:
            exposure = np.ones_like(y).astype('f')
        check_lengths(y, exposure)

        # normalize response
        y = y / exposure

        if weights is not None:
            weights = np.array(weights).astype('f')
        else:
            weights = np.ones_like(y).astype('f')
        check_lengths(weights, exposure)

        # set exposure as the weight
        weights = weights * exposure

        return y, weights

    def fit(self, X, y, exposure=None, weights=None):
        """Fit the generalized additive model.

        Parameters
        ----------
        X : array-like, shape (n_samples, m_features)
            Training vectors, where n_samples is the number of samples
            and m_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values (integers in classification, real numbers in
            regression)
            For classification, labels must correspond to classes.

        exposure : array-like shape (n_samples,) or None, default: None
            containing exposures
            if None, defaults to array of ones

        weights : array-like shape (n_samples,) or None, default: None
            containing sample weights
            if None, defaults to array of ones

        Returns
        -------
        self : object
            Returns fitted GAM object
        """
        y, weights = self._exposure_to_weights(y, exposure, weights)
        return super(PoissonGAM, self).fit(X, y, weights)

    def predict(self, X, exposure=None):
        """
        preduct expected value of target given model and input X
        often this is done via expected value of GAM given input X

        Parameters
        ---------
        X : array-like of shape (n_samples, m_features), default: None
            containing the input dataset

        exposure : array-like shape (n_samples,) or None, default: None
            containing exposures
            if None, defaults to array of ones

        Returns
        -------
        y : np.array of shape (n_samples,)
            containing predicted values under the model
        """
        if not self._is_fitted:
            raise AttributeError('GAM has not been fitted. Call fit first.')

        X = check_X(X, n_feats=len(self._n_coeffs) - self._fit_intercept,
                    edge_knots=self._edge_knots, dtypes=self._dtype)

        if exposure is not None:
            exposure = np.array(exposure).astype('f')
        else:
            exposure = np.ones(X.shape[0]).astype('f')
        check_lengths(X, exposure)

        return self.predict_mu(X) * exposure

    def gridsearch(self, X, y, exposure=None, weights=None,
                   return_scores=False, keep_best=True, objective='auto',
                   **param_grids):
        """
        performs a grid search over a space of parameters for a given objective

        NOTE:
        gridsearch method is lazy and will not remove useless combinations
        from the search space, eg.
          n_splines=np.arange(5,10), fit_splines=[True, False]
        will result in 10 loops, of which 5 are equivalent because
        even though fit_splines==False

        it is not recommended to search over a grid that alternates
        between known scales and unknown scales, as the scores of the
        cadidate models will not be comparable.

        Parameters
        ----------
        X : array
          input data of shape (n_samples, m_features)

        y : array
          label data of shape (n_samples,)

        exposure : array-like shape (n_samples,) or None, default: None
            containing exposures
            if None, defaults to array of ones

        weights : array-like shape (n_samples,) or None, default: None
            containing sample weights
            if None, defaults to array of ones

        return_scores : boolean, default False
          whether to return the hyperpamaters
          and score for each element in the grid

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
                objective scores as values
        else:
            self, ie possibly the newly fitted model
        """
        y, weights = self._exposure_to_weights(y, exposure, weights)
        return super(PoissonGAM, self).gridsearch(X, y,
                                                  weights=weights,
                                                  return_scores=return_scores,
                                                  keep_best=keep_best,
                                                  objective=objective,
                                                  **param_grids)


class GammaGAM(GAM):
    """Gamma GAM

    Parameters
    ----------
    callbacks : list of strings or list of CallBack objects,
                default: ['deviance', 'diffs']
        Names of callback objects to call during the optimization loop.

    constraints : str or callable, or iterable of str or callable,
                  default: None
        Names of constraint functions to call during the optimization loop.

        Must be in {'convex', 'concave', 'monotonic_inc', 'monotonic_dec',
                    'circular', 'none'}

        If None, then the model will apply no constraints.

        If only one str or callable is specified, then is it copied for all
        features.

    dtype : str in {'auto', 'numerical',  'categorical'},
            or list of str, default: 'auto'
        String describing the data-type of each feature.

        'numerical' is used for continuous-valued data-types,
            like in regression.
        'categorical' is used for discrete-valued data-types,
            like in classification.

        If only one str is specified, then is is copied for all features.

    lam : float or iterable of floats > 0, default: 0.6
        Smoothing strength; must be a positive float, or one positive float
        per feature.

        Larger values enforce stronger smoothing.

        If only one float is specified, then it is copied for all features.

    fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

        NOTE: the intercept receives no smoothing penalty.

    fit_linear : bool or iterable of bools, default: False
        Specifies if a linear term should be added to any of the feature
        functions. Useful for including pre-defined feature transformations
        in the model.

        If only one bool is specified, then it is copied for all features.

        NOTE: Many constraints are incompatible with an additional linear fit.
            eg. if a non-zero linear function is added to a periodic spline
            function, it will cease to be periodic.

            this is also possible for a monotonic spline function.

    fit_splines : bool or iterable of bools, default: True
        Specifies if a smoother should be added to any of the feature
        functions. Useful for defining feature transformations a-priori
        that should not have splines fitted to them.

        If only one bool is specified, then it is copied for all features.

        NOTE: fit_splines supercedes n_splines.
        ie. if n_splines > 0 and fit_splines = False, no splines will be fitted.

    max_iter : int, default: 100
        Maximum number of iterations allowed for the solver to converge.

    penalties : str or callable, or iterable of str or callable,
                default: 'auto'
        Type of penalty to use for each feature.

        penalty should be in {'auto', 'none', 'derivative', 'l2', }

        If 'auto', then the model will use 2nd derivative smoothing for features
        of dtype 'numerical', and L2 smoothing for features of dtype
        'categorical'.

        If only one str or callable is specified, then is it copied for all
        features.

    n_splines : int, or iterable of ints, default: 25
        Number of splines to use in each feature function; must be non-negative.
        If only one int is specified, then it is copied for all features.

        Note: this value is set to 0 if fit_splines is False

    scale : float or None, default: None
        scale of the distribution, if known a-priori.
        if None, scale is estimated.

    spline_order : int, or iterable of ints, default: 3
        Order of spline to use in each feature function; must be non-negative.
        If only one int is specified, then it is copied for all features

        Note: if a feature is of type categorical, spline_order will be set to 0.

    tol : float, default: 1e-4
        Tolerance for stopping criteria.

    Attributes
    ----------
    coef_ : array, shape (n_classes, m_features)
        Coefficient of the features in the decision function.
        If fit_intercept is True, then self.coef_[0] will contain the bias.

    statistics_ : dict
        Dictionary containing model statistics like GCV/UBRE scores, AIC/c,
        parameter covariances, estimated degrees of freedom, etc.

    logs_ : dict
        Dictionary containing the outputs of any callbacks at each
        optimization loop.

        The logs are structured as `{callback: [...]}`

    References
    ----------
    Simon N. Wood, 2006
    Generalized Additive Models: an introduction with R

    Hastie, Tibshirani, Friedman
    The Elements of Statistical Learning
    http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf

    Paul Eilers & Brian Marx, 2015
    International Biometric Society: A Crash Course on P-splines
    http://www.ibschannel2015.nl/project/userfiles/Crash_course_handout.pdf
    """
    def __init__(self, lam=0.6, max_iter=100, n_splines=25, spline_order=3,
                 penalties='auto', dtype='auto', tol=1e-4, scale=None,
                 callbacks=['deviance', 'diffs'],
                 fit_intercept=True, fit_linear=False, fit_splines=True,
                 constraints=None):
        self.scale = scale
        super(GammaGAM, self).__init__(distribution=GammaDist(scale=self.scale),
                                        link='inverse',
                                        lam=lam,
                                        dtype=dtype,
                                        max_iter=max_iter,
                                        n_splines=n_splines,
                                        spline_order=spline_order,
                                        penalties=penalties,
                                        tol=tol,
                                        callbacks=callbacks,
                                        fit_intercept=fit_intercept,
                                        fit_linear=fit_linear,
                                        fit_splines=fit_splines,
                                        constraints=constraints)

        self._exclude += ['distribution', 'link']

    def _validate_params(self):
        """
        method to sanitize model parameters

        Parameters
        ---------
        None

        Returns
        -------
        None
        """
        self.distribution = GammaDist(scale=self.scale)
        super(GammaGAM, self)._validate_params()


class InvGaussGAM(GAM):
    """Inverse Gaussian GAM

    Parameters
    ----------
    callbacks : list of strings or list of CallBack objects,
                default: ['deviance', 'diffs']
        Names of callback objects to call during the optimization loop.

    constraints : str or callable, or iterable of str or callable,
                  default: None
        Names of constraint functions to call during the optimization loop.

        Must be in {'convex', 'concave', 'monotonic_inc', 'monotonic_dec',
                    'circular', 'none'}

        If None, then the model will apply no constraints.

        If only one str or callable is specified, then is it copied for all
        features.

    dtype : str in {'auto', 'numerical',  'categorical'},
            or list of str, default: 'auto'
        String describing the data-type of each feature.

        'numerical' is used for continuous-valued data-types,
            like in regression.
        'categorical' is used for discrete-valued data-types,
            like in classification.

        If only one str is specified, then is is copied for all features.

    lam : float or iterable of floats > 0, default: 0.6
        Smoothing strength; must be a positive float, or one positive float
        per feature.

        Larger values enforce stronger smoothing.

        If only one float is specified, then it is copied for all features.

    fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

        NOTE: the intercept receives no smoothing penalty.

    fit_linear : bool or iterable of bools, default: False
        Specifies if a linear term should be added to any of the feature
        functions. Useful for including pre-defined feature transformations
        in the model.

        If only one bool is specified, then it is copied for all features.

        NOTE: Many constraints are incompatible with an additional linear fit.
            eg. if a non-zero linear function is added to a periodic spline
            function, it will cease to be periodic.

            this is also possible for a monotonic spline function.

    fit_splines : bool or iterable of bools, default: True
        Specifies if a smoother should be added to any of the feature
        functions. Useful for defining feature transformations a-priori
        that should not have splines fitted to them.

        If only one bool is specified, then it is copied for all features.

        NOTE: fit_splines supercedes n_splines.
        ie. if n_splines > 0 and fit_splines = False, no splines will be fitted.

    max_iter : int, default: 100
        Maximum number of iterations allowed for the solver to converge.

    penalties : str or callable, or iterable of str or callable,
                default: 'auto'
        Type of penalty to use for each feature.

        penalty should be in {'auto', 'none', 'derivative', 'l2', }

        If 'auto', then the model will use 2nd derivative smoothing for features
        of dtype 'numerical', and L2 smoothing for features of dtype
        'categorical'.

        If only one str or callable is specified, then is it copied for all
        features.

    n_splines : int, or iterable of ints, default: 25
        Number of splines to use in each feature function; must be non-negative.
        If only one int is specified, then it is copied for all features.

        Note: this value is set to 0 if fit_splines is False

    scale : float or None, default: None
        scale of the distribution, if known a-priori.
        if None, scale is estimated.

    spline_order : int, or iterable of ints, default: 3
        Order of spline to use in each feature function; must be non-negative.
        If only one int is specified, then it is copied for all features

        Note: if a feature is of type categorical, spline_order will be set to 0.

    tol : float, default: 1e-4
        Tolerance for stopping criteria.

    Attributes
    ----------
    coef_ : array, shape (n_classes, m_features)
        Coefficient of the features in the decision function.
        If fit_intercept is True, then self.coef_[0] will contain the bias.

    statistics_ : dict
        Dictionary containing model statistics like GCV/UBRE scores, AIC/c,
        parameter covariances, estimated degrees of freedom, etc.

    logs_ : dict
        Dictionary containing the outputs of any callbacks at each
        optimization loop.

        The logs are structured as `{callback: [...]}`

    References
    ----------
    Simon N. Wood, 2006
    Generalized Additive Models: an introduction with R

    Hastie, Tibshirani, Friedman
    The Elements of Statistical Learning
    http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf

    Paul Eilers & Brian Marx, 2015
    International Biometric Society: A Crash Course on P-splines
    http://www.ibschannel2015.nl/project/userfiles/Crash_course_handout.pdf
    """
    def __init__(self, lam=0.6, max_iter=100, n_splines=25, spline_order=3,
                 penalties='auto', dtype='auto', tol=1e-4, scale=None,
                 callbacks=['deviance', 'diffs'],
                 fit_intercept=True, fit_linear=False, fit_splines=True,
                 constraints=None):
        self.scale = scale
        super(InvGaussGAM, self).__init__(distribution=InvGaussDist(scale=self.scale),
                                          link='inv_squared',
                                          lam=lam,
                                          dtype=dtype,
                                          max_iter=max_iter,
                                          n_splines=n_splines,
                                          spline_order=spline_order,
                                          penalties=penalties,
                                          tol=tol,
                                          callbacks=callbacks,
                                          fit_intercept=fit_intercept,
                                          fit_linear=fit_linear,
                                          fit_splines=fit_splines,
                                          constraints=constraints)

        self._exclude += ['distribution', 'link']

    def _validate_params(self):
        """
        method to sanitize model parameters

        Parameters
        ---------
        None

        Returns
        -------
        None
        """
        self.distribution = InvGaussDist(scale=self.scale)
        super(InvGaussGAM, self)._validate_params()
