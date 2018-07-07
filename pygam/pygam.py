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
from pygam.utils import make_2d
from pygam.utils import check_array
from pygam.utils import check_lengths
from pygam.utils import load_diagonal
from pygam.utils import TablePrinter
from pygam.utils import space_row
from pygam.utils import sig_code
from pygam.utils import gen_edge_knots
from pygam.utils import b_spline_basis
from pygam.utils import combine
from pygam.utils import cholesky
from pygam.utils import check_param
from pygam.utils import isiterable
from pygam.utils import NotPositiveDefiniteError
from pygam.utils import OptimizationError


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

    verbose : bool, default: False
        whether to show pyGAM warnings

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
                 dtype='auto', constraints=None, verbose=False):

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
        self.verbose = verbose

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
                if dt == 'categorical' and self.verbose:
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
                if fl and self.verbose:
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
        self._edge_knots = [gen_edge_knots(feat, dtype, verbose=self.verbose) for feat, dtype in \
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

    def generate_X_grid(self, n=500):
        """create a nice grid of X data

        array is sorted by feature and uniformly spaced,
        so the marginal and joint distributions are likely wrong

        Parameters
        ----------
        n : int, default: 500
            number of data points to create

        Returns
        -------
        np.array of shape (n, n_features)
        """
        if not self._is_fitted:
            raise AttributeError('GAM has not been fitted. Call fit first.')
        X = []
        for ek in self._edge_knots:
            X.append(np.linspace(ek[0], ek[-1], num=n))
        return np.vstack(X).T

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
        y = check_y(y, self.link, self.distribution, verbose=self.verbose)
        mu = self.predict_mu(X)

        if weights is not None:
            weights = np.array(weights).astype('f').ravel()
            weights = check_array(weights, name='sample weights',
                                  ndim=1, verbose=self.verbose)
            check_lengths(y, weights)
        else:
            weights = np.ones_like(y).astype('float64')

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
        return self.distribution.log_pdf(y=y, mu=mu, weights=weights).sum()

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
                    edge_knots=self._edge_knots, dtypes=self._dtype,
                    verbose=self.verbose)

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
                    edge_knots=self._edge_knots, dtypes=self._dtype,
                    verbose=self.verbose)

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
        modelmat : sparse matrix of len n_samples
            containing model matrix of the spline basis for selected features
        """
        X = check_X(X, n_feats=len(self._n_coeffs) - self._fit_intercept,
                    edge_knots=self._edge_knots, dtypes=self._dtype,
                    verbose=self.verbose)

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
                                             sparse=True,
                                             verbose=self.verbose))

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
                L = cholesky(A, verbose=self.verbose, **kwargs)
                self._constraint_l2 = constraint_l2
                return L
            except NotPositiveDefiniteError:
                if self.verbose:
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
        if mask.sum() == 0:
            raise OptimizationError('PIRLS optimization has diverged.\n' +
                'Try increasing regularization, or specifying an initial value for self.coef_')
        return mask


    def _initial_estimate(self, y, modelmat):
        """
        Makes an inital estimate for the model coefficients.

        For a LinearGAM we simply initialize to small coefficients.

        For other GAMs we transform the problem to the linear space
        and solve an unpenalized version.

        Parameters
        ---------
        y : array-like of shape (n,)
            containing target data
        modelmat : sparse matrix of shape (n, m)
            containing model matrix of the spline basis

        Returns
        -------
        coef : array of shape (m,) containing the initial estimate for the model
            coefficients

        Notes
        -----
        This method implements the suggestions in
            Wood, section 2.2.2 Geometry and IRLS convergence, pg 80
        """

        # do a simple initialization for LinearGAMs
        if isinstance(self, LinearGAM):
            n, m = modelmat.shape
            return np.ones(m) * np.sqrt(EPS)

        # transform the problem to the linear scale
        y = deepcopy(y).astype('float64')
        y[y == 0] += .01 # edge case for log link, inverse link, and logit link
        y[y == 1] -= .01 # edge case for logit link

        y_ = self.link.link(y, self.distribution)
        y_ = make_2d(y_, verbose=False)
        assert np.isfinite(y_).all(), "transformed response values should be well-behaved."

        # solve the linear problem
        modelmat = modelmat.A
        return np.linalg.solve(load_diagonal(modelmat.T.dot(modelmat)),
                               modelmat.T.dot(y_))

        # not sure if this is faster...
        # return np.linalg.pinv(modelmat.T.dot(modelmat)).dot(modelmat.T.dot(y_))

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

        # initialize GLM coefficients if model is not yet fitted
        if (not self._is_fitted or
            len(self.coef_) != sum(self._n_coeffs) or
            not np.isfinite(self.coef_).all()):

           # initialize the model
           self.coef_ = self._initial_estimate(Y, modelmat)

        assert np.isfinite(self.coef_).all(), "coefficients should be well-behaved, but found: {}".format(self.coef_)

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

        min_n_m = np.min([m,n])
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

            # need to recompute the number of singular values
            min_n_m = np.min([m, n, mask.sum()])
            Dinv = np.zeros((min_n_m + m, m)).T

            # SVD
            U, d, Vt = np.linalg.svd(np.vstack([R, E.T]))
            svd_mask = d <= (d.max() * np.sqrt(EPS)) # mask out small singular values

            np.fill_diagonal(Dinv, d**-1) # invert the singular values
            U1 = U[:min_n_m,:] # keep only top portion of U

            # update coefficients
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
        y = check_y(y, self.link, self.distribution, verbose=self.verbose)
        X = check_X(X, verbose=self.verbose)
        check_X_y(X, y)

        if weights is not None:
            weights = np.array(weights).astype('f').ravel()
            weights = check_array(weights, name='sample weights',
                                  ndim=1, verbose=self.verbose)
            check_lengths(y, weights)
        else:
            weights = np.ones_like(y).astype('float64')

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

        y = check_y(y, self.link, self.distribution, verbose=self.verbose)
        X = check_X(X, n_feats=len(self._n_coeffs) - self._fit_intercept,
                    edge_knots=self._edge_knots, dtypes=self._dtype,
                    verbose=self.verbose)
        check_X_y(X, y)

        if weights is not None:
            weights = np.array(weights).astype('f').ravel()
            weights = check_array(weights, name='sample weights',
                                  ndim=1, verbose=self.verbose)
            check_lengths(y, weights)
        else:
            weights = np.ones_like(y).astype('float64')

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
        - se: standarrd errors
        - AIC: Akaike Information Criterion
        - AICc: corrected Akaike Information Criterion
        - pseudo_r2: dict of Pseudo R-squared metrics
        - GCV: generailized cross-validation
            or
        - UBRE: Un-Biased Risk Estimator
        - n_samples: number of samples used in estimation

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
        self.statistics_['n_samples'] = len(y)
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
        self.statistics_['p_values'] = self._estimate_p_values()

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
            weights = np.ones_like(y).astype('float64')

        null_mu = y.mean() * np.ones_like(y).astype('float64')

        null_d = self.distribution.deviance(y=y, mu=null_mu, weights=weights)
        full_d = self.distribution.deviance(y=y, mu=mu, weights=weights)

        null_ll = self._loglikelihood(y=y, mu=null_mu, weights=weights)
        full_ll = self._loglikelihood(y=y, mu=mu, weights=weights)

        r2 = OrderedDict()
        r2['explained_deviance'] = 1. - full_d.sum()/null_d.sum()
        r2['McFadden'] = full_ll/null_ll
        r2['McFadden_adj'] = 1. - (full_ll - self.statistics_['edof'])/null_ll

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
            weights = np.ones_like(y).astype('float64')

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

    def _estimate_p_values(self):
        """estimate the p-values for all features
        """
        if not self._is_fitted:
            raise AttributeError('GAM has not been fitted. Call fit first.')

        p_values = []
        for feature in range(len(self._n_coeffs)):
            p_values.append(self._compute_p_value(feature))

        return p_values

    def _compute_p_value(self, feature):
        """compute the p-value of the desired feature

        Arguments
        ---------
        feature : int
            feature to select from the data.
            when fit_intercept=True, 0 corresponds to the intercept

        Returns
        -------
        p_value : float

        Notes
        -----
        Wood 2006, section 4.8.5:
            The p-values, calculated in this manner, behave correctly for un-penalized models,
            or models with known smoothing parameters, but when smoothing parameters have
            been estimated, the p-values are typically lower than they should be, meaning that
            the tests reject the null too readily.

                (...)

            In practical terms, if these p-values suggest that a term is not needed in a model,
            then this is probably true, but if a term is deemed ‘significant’ it is important to be
            aware that this significance may be overstated.

        based on equations from Wood 2006 section 4.8.5 page 191
        and errata https://people.maths.bris.ac.uk/~sw15190/igam/iGAMerrata-12.pdf

        the errata shows a correction for the f-statisitc.
        """
        if not self._is_fitted:
            raise AttributeError('GAM has not been fitted. Call fit first.')

        idxs = self._select_feature(feature)
        cov = self.statistics_['cov'][idxs][:, idxs]
        coef = self.coef_[idxs]

        # center non-intercept feature functions
        if feature > 0 or self.fit_intercept is False:
            fit_linear = self._fit_linear[feature - self.fit_intercept]
            n_splines = self._n_splines[feature - self.fit_intercept]

            # only do this if we even have splines
            if n_splines > 0:
                coef[fit_linear:]-= coef[fit_linear:].mean()

        inv_cov, rank = sp.linalg.pinv(cov, return_rank=True)
        score = coef.T.dot(inv_cov).dot(coef)

        # compute p-values
        if self.distribution._known_scale:
            # for known scale use chi-squared statistic
            return 1 - sp.stats.chi2.cdf(x=score, df=rank)
        else:
            # if scale has been estimated, prefer to use f-statisitc
            score = score / rank
            return 1 - sp.stats.f.cdf(score, rank, self.statistics_['n_samples'] - self.statistics_['edof'])

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


        Notes
        -----
        Wood 2006, section 4.9
            Confidence intervals based on section 4.8 rely on large sample results to deal with
            non-Gaussian distributions, and treat the smoothing parameters as fixed, when in
            reality they are estimated from the data.
        """
        if not self._is_fitted:
            raise AttributeError('GAM has not been fitted. Call fit first.')

        X = check_X(X, n_feats=len(self._n_coeffs) - self._fit_intercept,
                    edge_knots=self._edge_knots, dtypes=self._dtype,
                    verbose=self.verbose)

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

        Notes
        -----
        when the scale parameter is known, then we can proceed with a large
        sample approximation to the distribution of the model coefficients
        where B_hat ~ Normal(B, cov)

        when the scale parameter is unknown, then we have to account for
        the distribution of the estimated scale parameter, which is Chi-squared.
        since we scale our estimate of B_hat by the sqrt of estimated scale,
        we get a t distribution: Normal / sqrt(Chi-squared) ~ t

        see Simon Wood section 1.3.2, 1.3.3, 1.5.5, 2.1.5
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
        cov = self.statistics_['cov'][idxs][:, idxs]

        var = (modelmat.dot(cov) * modelmat.todense().A).sum(axis=1)
        if prediction:
            var += self.distribution.scale

        lines = []
        for quantile in quantiles:
            if self.distribution._known_scale:
                q = sp.stats.norm.ppf(quantile)
            else:
                q = sp.stats.t.ppf(quantile, df=self.statistics_['n_samples'] -
                                                self.statistics_['edof'])

            lines.append(lp + q * var**0.5)
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
            raise ValueError('feature {} out of range for {}-dimensional data'\
                             .format(feature, len(self._n_splines)))

        if feature == -1:
            # special case for selecting all features
            return np.arange(np.sum(self._n_coeffs), dtype=int)

        a = np.sum(self._n_coeffs[:feature])
        b = np.sum(self._n_coeffs[feature])
        return np.arange(a, a+b, dtype=int)

    def partial_dependence(self, X=None, feature=-1, width=None, quantiles=None):
        """
        Computes the feature functions for the GAM
        and possibly their confidence intervals.

        if both width=None and quantiles=None,
        then no confidence intervals are computed

        Parameters
        ----------
        X : array or None, default: None
            input data of shape (n_samples, m_features).
            if None, an equally spaced grid of 500 points is generated for
            each feature function.
        feature : array-like of ints, default: -1
            feature for which to compute the partial dependence functions
            if feature == -1, then all features are selected,
            excluding the intercept
            if feature == 'intercept' and gam.fit_intercept is True,
            then the intercept's partial dependence is returned
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

        if X is not None:
            X = check_X(X, n_feats=m,
                        edge_knots=self._edge_knots, dtypes=self._dtype,
                        verbose=self.verbose)
        else:
            X = self.generate_X_grid()

        p_deps = []

        compute_quantiles = (width is not None) or (quantiles is not None)
        conf_intervals = []

        # make coding more pythonic for users
        if feature == 'intercept':
            if not self._fit_intercept:
                raise ValueError('intercept is not fitted')
            feature = 0
        elif feature == -1:
            feature = np.arange(m) + self._fit_intercept
        else:
            feature += self._fit_intercept

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

        # high-level model summary
        width_details = 47
        width_results = 58

        model_fmt = [
            (self.__class__.__name__, 'model_details', width_details),
            ('', 'model_results', width_results)
            ]

        model_details = []

        if self.distribution._known_scale:
            objective = 'UBRE'
        else:
            objective = 'GCV'

        model_details.append({'model_details': space_row('Distribution:', self.distribution.__class__.__name__, total_width=width_details),
                              'model_results': space_row('Effective DoF:', str(np.round(self.statistics_['edof'], 4)), total_width=width_results)})
        model_details.append({'model_details': space_row('Link Function:', self.link.__class__.__name__, total_width=width_details),
                              'model_results': space_row('Log Likelihood:', str(np.round(self.statistics_['loglikelihood'], 4)), total_width=width_results)})
        model_details.append({'model_details': space_row('Number of Samples:', str(self.statistics_['n_samples']), total_width=width_details),
                              'model_results': space_row('AIC: ', str(np.round(self.statistics_['AIC'], 4)), total_width=width_results)})
        model_details.append({'model_results': space_row('AICc: ', str(np.round(self.statistics_['AICc'], 4)), total_width=width_results)})
        model_details.append({'model_results': space_row(objective + ':', str(np.round(self.statistics_[objective], 4)), total_width=width_results)})
        model_details.append({'model_results': space_row('Scale:', str(np.round(self.statistics_['scale'], 4)), total_width=width_results)})
        model_details.append({'model_results': space_row('Pseudo R-Squared:', str(np.round(self.statistics_['pseudo_r2']['explained_deviance'], 4)), total_width=width_results)})

        # feature summary
        data = []

        for i in np.arange(len(self._n_splines)):
            data.append({
                'feature_func': 'feature {}'.format(i  + self.fit_intercept),
                'n_splines': self._n_splines[i],
                'spline_order': self._spline_order[i],
                'fit_linear': self._fit_linear[i],
                'dtype': self._dtype[i],
                'lam': np.round(self._lam[i + self.fit_intercept], 4),
                'p_value': '%.2e'%(self.statistics_['p_values'][i  + self.fit_intercept]),
                'sig_code': sig_code(self.statistics_['p_values'][i  + self.fit_intercept])
            })

        if self.fit_intercept:
            data.append({
                    'feature_func': 'intercept',
                    'n_splines': '',
                    'spline_order': '',
                    'fit_linear': '',
                    'dtype': '',
                    'lam': '',
                    'p_value': '%.2e'%(self.statistics_['p_values'][0]),
                    'sig_code': sig_code(self.statistics_['p_values'][0])
                })

        fmt = [
            ('Feature Function',          'feature_func',          18),
            ('Data Type',          'dtype',          14),
            ('Num Splines',          'n_splines',          13),
            ('Spline Order',          'spline_order',       13),
            ('Linear Fit',          'fit_linear',          11),
            ('Lambda',          'lam',           10),
            ('P > x',          'p_value',          10),
            ('Sig. Code',          'sig_code',          10)
            ]

        print( TablePrinter(model_fmt, ul='=', sep=' ')(model_details) )
        print("="*106)
        print( TablePrinter(fmt, ul='=')(data) )
        print("="*106)
        print("Significance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        print()
        print("WARNING: Fitting splines and a linear function to a feature introduces a model identifiability problem\n" \
              "         which can cause p-values to appear significant when they are not.")
        print()
        print("WARNING: p-values calculated in this manner behave correctly for un-penalized models or models with\n" \
              "         known smoothing parameters, but when smoothing parameters have been estimated, the p-values\n" \
              "         are typically lower than they should be, meaning that the tests reject the null too readily.")

    def gridsearch(self, X, y, weights=None, return_scores=False,
                   keep_best=True, objective='auto', progress=True,
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

        progress : bool, default: True
            whether to display a progress bar

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
        # check if model fitted
        if not self._is_fitted:
            self._validate_params()

        y = check_y(y, self.link, self.distribution, verbose=self.verbose)
        X = check_X(X, verbose=self.verbose)
        check_X_y(X, y)

        if weights is not None:
            weights = np.array(weights).astype('f').ravel()
            weights = check_array(weights, name='sample weights',
                                  ndim=1, verbose=self.verbose)
            check_lengths(y, weights)
        else:
            weights = np.ones_like(y).astype('float64')

        # validate objective
        if objective not in ['auto', 'GCV', 'UBRE', 'AIC', 'AICc']:
            raise ValueError("objective mut be in "\
                             "['auto', 'GCV', 'UBRE', 'AIC', 'AICc'], '\
                             'but found objective = {}".format(objective))

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

        # make progressbar optional
        if progress:
            pbar = ProgressBar()
        else:
            pbar = lambda x: x

        # loop through candidate model params
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
                if self.verbose:
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
            if self.verbose:
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

    def sample(self, X, y, quantity='y', sample_at_X=None,
               weights=None, n_draws=100, n_bootstraps=1, objective='auto'):
        """Simulate from the posterior of the coefficients and smoothing params.

        Samples are drawn from the posterior of the coefficients and smoothing
        parameters given the response in an approximate way. The GAM must
        already be fitted before calling this method; if the model has not
        been fitted, then an exception is raised. Moreover, it is recommended
        that the model and its hyperparameters be chosen with `gridsearch`
        (with the parameter `keep_best=True`) before calling `sample`, so that
        the result of that gridsearch can be used to generate useful response
        data and so that the model's coefficients (and their covariance matrix)
        can be used as the first bootstrap sample.

        These samples are drawn as follows. Details are in the reference below.

        1. `n_bootstraps` many "bootstrap samples" of the response (`y`) are
        simulated by drawing random samples from the model's distribution
        evaluated at the expected values (`mu`) for each sample in `X`.
        2. A copy of the model is fitted to each of those bootstrap samples of
        the response. The result is an approximation of the distribution over
        the smoothing parameter `lam` given the response data `y`.
        3. Samples of the coefficients are simulated from a multivariate normal
        using the bootstrap samples of the coefficients and their covariance
        matrices.

        NOTE: A `gridsearch` is done `n_bootstraps` many times, so keep
        `n_bootstraps` small. Make `n_bootstraps < n_draws` to take advantage
        of the expensive bootstrap samples of the smoothing parameters.

        NOTE: For now, the grid of `lam` values is the default of `gridsearch`.
        Until randomized grid search is implemented, it is not worth setting
        `n_bootstraps` to a value greater than one because the smoothing
        parameters will be identical in each bootstrap sample.

        Parameters
        -----------
        X : array of shape (n_samples, m_features)
              empirical input data

        y : array of shape (n_samples,)
              empirical response vector

        quantity : {'y', 'coef', 'mu'}, default: 'y'
            What quantity to return pseudorandom samples of.
            If `sample_at_X` is not None and `quantity` is either `'y'` or
            `'mu'`, then samples are drawn at the values of `X` specified in
            `sample_at_X`.

        sample_at_X : array of shape (n_samples_to_simulate, m_features) or
        None, default: None
            Input data at which to draw new samples.

            Only applies for `quantity` equal to `'y'` or to `'mu`'.
            If `None`, then `sample_at_X` is replaced by `X`.

        weights : np.array of shape (n_samples,)
            sample weights

        n_draws : positive int, default: 100
            The number of samples to draw from the posterior distribution of
            the coefficients and smoothing parameters

        n_bootstraps : positive int, default: 1
            The number of bootstrap samples to draw from simulations of the
            response (from the already fitted model) to estimate the
            distribution of the smoothing parameters given the response data.
            If `n_bootstraps` is 1, then only the already fitted model's
            smoothing parameter is used, and the distribution over the
            smoothing parameters is not estimated using bootstrap sampling.

        objective : string, default: 'auto'
            metric to optimize in grid search. must be in
            ['AIC', 'AICc', 'GCV', 'UBRE', 'auto']
            if 'auto', then grid search will optimize GCV for models with
            unknown scale and UBRE for models with known scale.

        Returns
        -------
        draws : 2D array of length n_draws
            Simulations of the given `quantity` using samples from the
            posterior distribution of the coefficients and smoothing parameter
            given the response data. Each row is a pseudorandom sample.

            If `quantity == 'coef'`, then the number of columns of `draws` is
            the number of coefficients (`len(self.coef_)`).

            Otherwise, the number of columns of `draws` is the number of
            rows of `sample_at_X` if `sample_at_X` is not `None` or else
            the number of rows of `X`.

        References
        ----------
        Simon N. Wood, 2006. Generalized Additive Models: an introduction with
        R. Section 4.9.3 (pages 198–199) and Section 5.4.2 (page 256–257).
        """
        if quantity not in {'mu', 'coef', 'y'}:
            raise ValueError("`quantity` must be one of 'mu', 'coef', 'y';"
                             " got {}".format(quantity))

        coef_draws = self._sample_coef(
            X, y, weights=weights, n_draws=n_draws,
            n_bootstraps=n_bootstraps, objective=objective)
        if quantity == 'coef':
            return coef_draws

        if sample_at_X is None:
            sample_at_X = X

        linear_predictor = self._modelmat(sample_at_X).dot(coef_draws.T)
        mu_shape_n_draws_by_n_samples = self.link.mu(
            linear_predictor, self.distribution).T
        if quantity == 'mu':
            return mu_shape_n_draws_by_n_samples
        else:
            return self.distribution.sample(mu_shape_n_draws_by_n_samples)

    def _sample_coef(self, X, y, weights=None, n_draws=100, n_bootstraps=1,
                     objective='auto'):
        """Simulate from the posterior of the coefficients.

        NOTE: A `gridsearch` is done `n_bootstraps` many times, so keep
        `n_bootstraps` small. Make `n_bootstraps < n_draws` to take advantage
        of the expensive bootstrap samples of the smoothing parameters.

        For now, the grid of `lam` values is the default of `gridsearch`.

        Parameters
        -----------
        X : array of shape (n_samples, m_features)
              input data

        y : array of shape (n_samples,)
              response vector

        weights : np.array of shape (n_samples,)
            sample weights

        n_draws : positive int, default: 100
            The number of samples to draw from the posterior distribution of
            the coefficients and smoothing parameters

        n_bootstraps : positive int, default: 1
            The number of bootstrap samples to draw from simulations of the
            response (from the already fitted model) to estimate the
            distribution of the smoothing parameters given the response data.
            If `n_bootstraps` is 1, then only the already fitted model's
            smoothing parameters is used.

        objective : string, default: 'auto'
            metric to optimize in grid search. must be in
            ['AIC', 'AICc', 'GCV', 'UBRE', 'auto']
            if 'auto', then grid search will optimize GCV for models with
            unknown scale and UBRE for models with known scale.

        Returns
        -------
        coef_samples : array of shape (n_draws, n_samples)
            Approximate simulations of the coefficients drawn from the
            posterior distribution of the coefficients and smoothing
            parameters given the response data

        References
        ----------
        Simon N. Wood, 2006. Generalized Additive Models: an introduction with
        R. Section 4.9.3 (pages 198–199) and Section 5.4.2 (page 256–257).
        """
        if not self._is_fitted:
            raise AttributeError('GAM has not been fitted. Call fit first.')
        if n_bootstraps < 1:
            raise ValueError('n_bootstraps must be >= 1;'
                             ' got {}'.format(n_bootstraps))
        if n_draws < 1:
            raise ValueError('n_draws must be >= 1;'
                             ' got {}'.format(n_draws))

        coef_bootstraps, cov_bootstraps = (
            self._bootstrap_samples_of_smoothing(X, y, weights=weights,
                                                 n_bootstraps=n_bootstraps,
                                                 objective=objective))
        coef_draws = self._simulate_coef_from_bootstraps(
            n_draws, coef_bootstraps, cov_bootstraps)

        return coef_draws

    def _bootstrap_samples_of_smoothing(self, X, y, weights=None,
                                        n_bootstraps=1, objective='auto'):
        """Sample the smoothing parameters using simulated response data."""
        mu = self.predict_mu(X)  # Wood pg. 198 step 1
        coef_bootstraps = [self.coef_]
        cov_bootstraps = [
            load_diagonal(self.statistics_['cov'])]

        for _ in range(n_bootstraps - 1):  # Wood pg. 198 step 2
            # generate response data from fitted model (Wood pg. 198 step 3)
            y_bootstrap = self.distribution.sample(mu)

            # fit smoothing parameters on the bootstrap data
            # (Wood pg. 198 step 4)
            # TODO: Either enable randomized searches over hyperparameters
            # (like in sklearn's RandomizedSearchCV), or draw enough samples of
            # `lam` so that each of these bootstrap samples get different
            # values of `lam`. Right now, each bootstrap sample uses the exact
            # same grid of values for `lam`, so it is not worth setting
            # `n_bootstraps > 1`.
            gam = deepcopy(self)
            gam.set_params(self.get_params())
            gam.gridsearch(X, y_bootstrap, weights=weights,
                           objective=objective)
            lam = gam.lam

            # fit coefficients on the original data given the smoothing params
            # (Wood pg. 199 step 5)
            gam = deepcopy(self)
            gam.set_params(self.get_params())
            gam.lam = lam
            gam.fit(X, y, weights=weights)

            coef_bootstraps.append(gam.coef_)

            cov = load_diagonal(gam.statistics_['cov'])

            cov_bootstraps.append(cov)
        return coef_bootstraps, cov_bootstraps

    def _simulate_coef_from_bootstraps(
            self, n_draws, coef_bootstraps, cov_bootstraps):
        """Simulate coefficients using bootstrap samples."""
        # Sample indices uniformly from {0, ..., n_bootstraps - 1}
        # (Wood pg. 199 step 6)
        random_bootstrap_indices = np.random.choice(
            np.arange(len(coef_bootstraps)), size=n_draws, replace=True)

        # Simulate `n_draws` many random coefficient vectors from a
        # multivariate normal distribution with mean and covariance given by
        # the bootstrap samples (indexed by `random_bootstrap_indices`) of
        # `coef_bootstraps` and `cov_bootstraps`. Because it's faster to draw
        # many samples from a certain distribution all at once, we make a dict
        # mapping bootstrap indices to draw indices and use the `size`
        # parameter of `np.random.multivariate_normal` to sample the draws
        # needed from that bootstrap sample all at once.
        bootstrap_index_to_draw_indices = defaultdict(list)
        for draw_index, bootstrap_index in enumerate(random_bootstrap_indices):
            bootstrap_index_to_draw_indices[bootstrap_index].append(draw_index)

        coef_draws = np.empty((n_draws, len(self.coef_)))

        for bootstrap, draw_indices in bootstrap_index_to_draw_indices.items():
            coef_draws[[draw_indices]] = np.random.multivariate_normal(
                coef_bootstraps[bootstrap], cov_bootstraps[bootstrap],
                size=len(draw_indices))

        return coef_draws


class LinearGAM(GAM):
    """Linear GAM

    This is a GAM with a Normal error distribution, and an identity link.

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

        NOTE: It is NOT recommended to use both 'fit_splines = True' and
        'fit_linear = True' for two reasons:
            (1) This introduces a model identifiabiilty problem, which can cause
            p-values to appear significant.

            (2) Many constraints are incompatible with an additional linear fit.
            eg. if a non-zero linear function is added to a periodic spline
            function, it will cease to be periodic.

            This is also possible for a monotonic spline function.

    fit_splines : bool or iterable of bools, default: True
        Specifies if a smoother should be added to any of the feature
        functions. Useful for defining feature transformations a-priori
        that should not have splines fitted to them.

        If only one bool is specified, then it is copied for all features.

        NOTE: fit_splines supercedes n_splines.
        ie. if n_splines > 0 and fit_splines = False, no splines will be fitted.

        NOTE: It is NOT recommended to use both 'fit_splines = True' and
        'fit_linear = True'.
        Please see 'fit_linear'

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

    verbose : bool, default: False
        whether to show pyGAM warnings

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
                 constraints=None, verbose=False):
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
                                        constraints=constraints,
                                        verbose=verbose)

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
                    edge_knots=self._edge_knots, dtypes=self._dtype,
                    verbose=self.verbose)

        return self._get_quantiles(X, width, quantiles, prediction=True)

class LogisticGAM(GAM):
    """Logistic GAM

    This is a GAM with a Binomial error distribution, and a logit link.

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

    verbose : bool, default: False
        whether to show pyGAM warnings

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
                 constraints=None, verbose=False):

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
                                          constraints=constraints,
                                          verbose=verbose)
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

        y = check_y(y, self.link, self.distribution, verbose=self.verbose)
        if X is not None:
            X = check_X(X, n_feats=len(self._n_coeffs) - self._fit_intercept,
                        edge_knots=self._edge_knots, dtypes=self._dtype,
                        verbose=self.verbose)

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

    This is a GAM with a Poisson error distribution, and a log link.

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

    verbose : bool, default: False
        whether to show pyGAM warnings

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
                 callbacks=['deviance', 'diffs'],
                 fit_intercept=True, fit_linear=False, fit_splines=True,
                 constraints=None, verbose=False):

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
                                         constraints=constraints,
                                         verbose=verbose)
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
            whether to scale the targets back up.
            useful when fitting with an exposure, in which case the count observations
            were scaled into rates. this rescales rates into counts.

        Returns
        -------
        log-likelihood : np.array of shape (n,)
            containing log-likelihood scores
        """
        if rescale_y:
            y = np.round(y * weights).astype('int')

        return self.distribution.log_pdf(y=y, mu=mu, weights=weights).sum()

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
        y = check_y(y, self.link, self.distribution, verbose=self.verbose)
        mu = self.predict_mu(X)

        if weights is not None:
            weights = np.array(weights).astype('f').ravel()
            weights = check_array(weights, name='sample weights',
                                  ndim=1, verbose=self.verbose)
            check_lengths(y, weights)
        else:
            weights = np.ones_like(y).astype('float64')

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
        y = y.ravel()

        if exposure is not None:
            exposure = np.array(exposure).astype('f').ravel()
            exposure = check_array(exposure, name='sample exposure',
                                   ndim=1, verbose=self.verbose)
        else:
            exposure = np.ones_like(y.ravel()).astype('float64')

        # check data
        exposure = exposure.ravel()
        check_lengths(y, exposure)

        # normalize response
        y = y / exposure

        if weights is not None:
            weights = np.array(weights).astype('f').ravel()
            weights = check_array(weights, name='sample weights',
                                  ndim=1, verbose=self.verbose)
        else:
            weights = np.ones_like(y).astype('float64')
        check_lengths(weights, exposure)

        # set exposure as the weight
        # we do this because we have divided our response
        # so if we make an error of 1 now, we need it to count more heavily.
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
                    edge_knots=self._edge_knots, dtypes=self._dtype,
                    verbose=self.verbose)

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

    This is a GAM with a Gamma error distribution, and a log link.

    NB
    Although canonical link function for the Gamma GLM is the inverse link,
    this function can create problems for numerical software because it becomes
    difficult to enforce the requirement that the mean of the Gamma distribution
    be positive. The log link guarantees this.

    If you need to use the inverse link function, simply construct a custom GAM:
    ```
    from pygam import GAM
    gam = GAM(distribution='gamma', link='inverse')
    ```

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

    verbose : bool, default: False
        whether to show pyGAM warnings

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
                 constraints=None, verbose=False):
        self.scale = scale
        super(GammaGAM, self).__init__(distribution=GammaDist(scale=self.scale),
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
                                        constraints=constraints,
                                        verbose=verbose)

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

    This is a GAM with a Inverse Gaussian error distribution, and a log link.

    NB
    Although canonical link function for the Inverse Gaussian GLM is the inverse squared link,
    this function can create problems for numerical software because it becomes
    difficult to enforce the requirement that the mean of the Inverse Gaussian distribution
    be positive. The log link guarantees this.

    If you need to use the inverse squared link function, simply construct a custom GAM:
    ```
    from pygam import GAM
    gam = GAM(distribution='inv_gauss', link='inv_squared')
    ```

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

    verbose : bool, default: False
        whether to show pyGAM warnings

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
                 constraints=None, verbose=False):
        self.scale = scale
        super(InvGaussGAM, self).__init__(distribution=InvGaussDist(scale=self.scale),
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
                                          constraints=constraints,
                                          verbose=verbose)

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
