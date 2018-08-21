"""
Link functions
"""
from __future__ import division, absolute_import
from abc import ABCMeta
from abc import abstractmethod, abstractproperty
from collections import defaultdict
import warnings
from copy import deepcopy

import numpy as np
import scipy as sp

from pygam.core import Core, nice_repr
from pygam.utils import isiterable, check_param, flatten, gen_edge_knots, b_spline_basis, tensor_product
from pygam.penalties import PENALTIES, CONSTRAINTS

DEFAULTS = {'lam': 0.6,
            'dtype': 'numerical',
            'fit_linear': False,
            'fit_splines': True,
            'penalties': 'auto',
            'constraints': None,
            'basis': 'ps',
            'by': None,
            'spline_order': 3,
            'n_splines': 20
            }

class Term(Core):
    __metaclass__ = ABCMeta
    def __init__(self, feature, lam=0.6, dtype='numerical',
                 fit_linear=False, fit_splines=True,
                 penalties='auto', constraints=None):
        """
        creates an instance of a Term object

        Parameters
        ----------

        Returns
        -------
        self
        """
        self.feature = feature

        self.lam = lam
        self.dtype = dtype
        self.fit_linear = fit_linear
        self.fit_splines = fit_splines
        self.penalties = penalties
        self.constraints = constraints

        if not(hasattr(self, '_name')):
            self._name = 'term'

        super(Term, self).__init__(name=self._name)
        self._defaults = self._get_defaults()
        self._validate_arguments()

    def __radd__(self, other):
        return TermList(other, self)

    def __add__(self, other):
        return TermList(self, other)

    def __mul__(self, other):
        raise NotImplementedError()

    def __repr__(self):
        """__repr__ method"""
        if hasattr(self, '_minimal_name'):
            name = self._minimal_name
        else:
            name = self.__class__.__name__

        features = [] if self.feature is None else self.feature
        features = np.atleast_1d(features).tolist()
        return nice_repr(name, {},
                         line_width=self._line_width,
                         line_offset=self._line_offset,
                         decimals=4, args=features)

    def _get_defaults(self):
        defaults = getattr(self, '_defaults', {})
        for k, v in self.get_params().items():
            if k in DEFAULTS and DEFAULTS[k] == v:
                defaults[k] = v
        return defaults

    def _validate_arguments(self):
        # dtype
        if self.dtype not in ['auto', 'numerical', 'categorical']:
            raise ValueError("dtype must be in ['auto', 'numerical', "\
                             "'categorical'], "\
                             "but found dtype = {}".format(self.dtype))

        # fit_linear XOR fit_splines
        if self.fit_linear == self.fit_splines:
            raise ValueError('term must have fit_linear XOR fit_splines, but found: '
                             'fit_linear= {}, fit_splines={}'.format(self.fit_linear, self.fit_splines))

        # penalties
        if not isiterable(self.penalties):
            self.penalties = [self.penalties]

        for i, p in enumerate(self.penalties):
            if not (hasattr(p, '__call__') or
                    (p in PENALTIES) or
                    (p is None)):
                raise ValueError("penalties must be callable or in "\
                                 "{}, but found {} for {}th penalty"\
                                 .format(list(PENALTIES.keys()), p, i))

        # check lams and distribute to penalites
        if not isiterable(self.lam):
            self.lam = [self.lam]

        for lam in self.lam:
            check_param(lam, param_name='lam', dtype='float', constraint='>= 0')

        if len(self.lam) == 1:
            self.lam = self.lam * len(self.penalties)

        if len(self.lam) != len(self.penalties):
            raise ValueError('expected 1 lam per penalty, but found '\
                             'lam = {}, penalties = {}'.format(self.lam, self.penalties))

        # constraints
        if not isiterable(self.constraints):
            self.constraints = [self.constraints]

        for i, c in enumerate(self.constraints):
            if not (hasattr(c, '__call__') or
                    (c in CONSTRAINTS) or
                    (c is None)):
                raise ValueError("constraints must be callable or in "\
                                 "{}, but found {} for {}th constraint"\
                                 .format(list(CONSTRAINTS.keys()), c, i))

    @property
    def istensor(self):
        return isinstance(self, TensorTerm)

    @property
    def isintercept(self):
        return isinstance(self, Intercept)

    @property
    def term_list(self):
        return TermList(self)

    @property
    def info(self):
        info = self.get_params()
        info.update({'term_type': self._name})
        return info

    @classmethod
    def build_from_info(cls, info):
        info == deepcopy(info)
        if 'term_type' in info:
            cls_ = TERMS[info.pop('term_type')]
        else:
            cls_ = cls
        return cls_(**info)

    @property
    def hasconstraint(self):
        return any([c is not None for c in self.constraints])

    @property
    @abstractproperty
    def n_coefs(self):
        pass

    @abstractmethod
    def compile(self, X, verbose=False):
        return self

    @abstractmethod
    def build_columns(self, X, verbose=False):
        pass

    def build_penalties(self, verbose=False):
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
        if self.isintercept:
            return np.array([[0.]])

        Ps = []
        for penalty, lam in zip(self.penalties, self.lam):
            if penalty == 'auto':
                if self.dtype == 'numerical':
                    if self._name == 'spline_term':
                        penalty = 'derivative'
                    else:
                        penalty = 'l2'
                if self.dtype == 'categorical':
                    penalty = 'l2'
            if penalty is None:
                penalty = 'none'
            if penalty in PENALTIES:
                penalty = PENALTIES[penalty]

            penalty = penalty(self.n_coefs, coef=None) # penalties dont need coef
            Ps.append(np.multiply(penalty, lam))
        return np.prod(Ps)

    def build_constraints(self, coef, constraint_lam, constraint_l2):
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
        if self.isintercept:
            return np.array([[0.]])

        for constraints in self.constraints:

            if constraint is None:
                constraint = 'none'
            if constraint in CONSTRAINTS:
                constraint = CONSTRAINTS[constraint]

            c = constraint(n, coef) * constraint_lam
            Cs.append(c)


        Cs = sp.sparse.block_diag(Cs)

        # improve condition
        if Cs.nnz > 0:
            Cs += sp.sparse.diags(constraint_l2 * np.ones(Cs.shape[0]))

        return Cs

class Intercept(Term):
    def __init__(self):
        """
        creates an instance of a Intercept term

        Parameters
        ----------
        None

        Returns
        -------
        self
        """
        self._name = 'intercept_term'
        self._minimal_name = 'intercept'

        super(Intercept, self).__init__(feature=None, fit_linear=False, fit_splines=False, lam=0, penalties=None, constraints=None)

        self._exclude += ['fit_splines', 'fit_linear', 'lam', 'penalties', 'constraints', 'feature', 'dtype']
        self._args = []

    def __repr__(self):
        return self._minimal_name

    def _validate_arguments(self):
        # constraints
        self.constraints = [None]

    @property
    def n_coefs(self):
        return 1

    def compile(self, X, verbose=False):
        return self

    def build_columns(self, X, verbose=False):
        return sp.sparse.csc_matrix(np.ones((len(X), 1)))


class LinearTerm(Term):
    def __init__(self, feature, lam=0.6, penalties='auto'):
        """
        creates an instance of a LinearTerm

        Parameters
        ----------
        None

        Returns
        -------
        self
        """
        self._name = 'linear_term'
        self._minimal_name = 'l'
        super(LinearTerm, self).__init__(feature=feature, lam=lam,
                                         penalties=penalties,
                                         constraints=None, dtype='numerical',
                                         fit_linear=True, fit_splines=False)
        self._exclude += ['fit_splines', 'fit_linear', 'dtype', 'constraints']

    @property
    def n_coefs(self):
        return 1

    def compile(self, X, verbose=False):
        if self.feature >= X.shape[1]:
            raise ValueError('term requires feature {}, '\
                             'but X has only {} dimensions'\
                             .format(self.feature, X.shape[1]))

        self.edge_knots_ = gen_edge_knots(X[:, self.feature],
                                          self.dtype,
                                          verbose=verbose)
        return self

    def build_columns(self, X, verbose=False):
        return sp.sparse.csc_matrix(X[:, self.feature][:, np.newaxis])


class SplineTerm(Term):
    def __init__(self, feature, n_splines=20, spline_order=3, lam=0.6,
                 penalties='auto', constraints=None, dtype='numerical', basis='ps', by=None):
        """
        creates an instance of a SplineTerm

        Parameters
        ----------
        None

        Returns
        -------
        self
        """
        if basis is not 'ps':
            raise NotImplementedError('no basis function: {}'.format(basis))
        self.basis = basis
        self.n_splines = n_splines
        self.spline_order = spline_order
        self.by = by
        self._name = 'spline_term'
        self._minimal_name = 's'

        super(SplineTerm, self).__init__(feature=feature,
                                         lam=lam,
                                         penalties=penalties,
                                         constraints=constraints,
                                         fit_linear=False,
                                         fit_splines=True,
                                         dtype=dtype)

        self._exclude += ['fit_linear', 'fit_splines']

    def _validate_arguments(self):
        super(SplineTerm, self)._validate_arguments()

        # n_splines
        self.n_splines = check_param(self.n_splines, param_name='n_splines',
                                     dtype='int', constraint='>= 0')

        # spline_order
        self.spline_order = check_param(self.spline_order,
                                        param_name='spline_order',
                                        dtype='int', constraint='>= 0')

        # n_splines + spline_order
        if not self.n_splines > self.spline_order:
            raise ValueError('n_splines must be > spline_order. '\
                             'found: n_splines = {} and spline_order = {}'\
                             .format(self.n_splines, self.spline_order))

        # by
        if self.by is not None:
            self.by = check_param(self.by,
                                  param_name='by',
                                  dtype='int', constraint='>= 0')

    @property
    def n_coefs(self):
        return self.n_splines

    def compile(self, X, verbose=False):
        if self.feature >= X.shape[1]:
            raise ValueError('term requires feature {}, '\
                             'but X has only {} dimensions'\
                             .format(self.feature, X.shape[1]))

        if self.by is not None and self.by >= X.shape[1]:
            raise ValueError('by variable requires feature {}, '\
                             'but X has only {} dimensions'\
                             .format(self.by, X.shape[1]))

        self.edge_knots_ = gen_edge_knots(X[:, self.feature],
                                          self.dtype,
                                          verbose=verbose)
        return self

    def build_columns(self, X, verbose=False):
        splines = b_spline_basis
        X[:, self.feature][:, np.newaxis]

        splines = b_spline_basis(X[:, self.feature],
                                 edge_knots=self.edge_knots_,
                                 spline_order=self.spline_order,
                                 n_splines=self.n_splines,
                                 sparse=True,
                                 verbose=verbose)

        if self.by is not None:
            splines = splines.multiply(X[:, self.by][:, np.newaxis])

        return splines


class FactorTerm(SplineTerm):
    def __init__(self, feature, lam=0.6, penalties='auto'):
        """
        creates an instance of a FactorTerm

        Parameters
        ----------
        None

        Returns
        -------
        self
        """
        super(FactorTerm, self).__init__(feature=feature,
                                         lam=lam,
                                         dtype='categorical',
                                         spline_order=0,
                                         penalties=penalties,
                                         by=None,
                                         constraints=None)
        self._name = 'factor_term'
        self._minimal_name = 'f'

        self.n_splines = None
        self._exclude += ['dtype', 'spline_order', 'by', 'n_splines', 'basis', 'constraints']

    def compile(self, X, verbose=False):
        super(FactorTerm, self).compile(X)

        self.n_splines = len(self.edge_knots_) - 1

        self.edge_knots_ = gen_edge_knots(X[:, self.feature],
                                          self.dtype,
                                          verbose=verbose)
        return self


class TensorTerm(SplineTerm):
    _N_SPLINES = 10 # default num splines

    def __init__(self, *args, **kwargs):
        """
        creates an instance of an IdentityLink object

        Parameters
        ----------
        by : is applied to the resulting tensor product spline
             and is not distributed to the marginal splines.

        Returns
        -------
        self
        """
        by = kwargs.pop('by', None)
        self.verbose = kwargs.pop('verbose', False)
        self._terms = self._parse_terms(args, **kwargs)

        feature = [term.feature for term in self._terms]
        super(TensorTerm, self).__init__(feature, by=by)

        self._name = 'tensor_term'
        self._minimal_name = 'te'

        self._exclude = [
         'dtype',
         'fit_linear',
         'fit_splines',
         'lam',
         'n_splines',
         'spline_order',
         'constraints',
         'penalties',
         'basis',
        ]
        for param in self._exclude:
            delattr(self, param)

        self._defaults = self._get_defaults()

    def _parse_terms(self, args, **kwargs):
        m = len(args)
        if m < 2:
            raise ValueError('TensorTerm requires at least 2 marginal terms')

        for k, v in kwargs.items():
            if isiterable(v):
                if len(v) != m:
                    raise ValueError('Expected {} to have length {}, but found {} = {}'\
                                    .format(k, m, k, v))
            else:
                kwargs[k] = [v] * m

        terms = []
        for i, arg in enumerate(np.atleast_1d(args)):
            if isinstance(arg, TensorTerm):
                raise ValueError('TensorTerm does not accept other TensorTerms. '\
                                 'Please build a flat TensorTerm instead of a nested one.')

            if isinstance(arg, Term):
                if self.verbose and kwargs:
                    warnings.warn('kwargs are skipped when Term instances are passed to TensorTerm constructor')
                terms.append(arg)
                continue

            kwargs_ = {'n_splines': self._N_SPLINES}
            kwargs_.update({k: v[i] for k, v in kwargs.items()})

            terms.append(SplineTerm(arg, **kwargs_))

        return terms

    def __len__(self):
        return len(self._terms)

    def __getitem__(self, i):
            return self._terms[i]

    @property
    def info(self):
        info = super(TensorTerm, self).info
        info.update({'terms':[term.info for term in self._terms]})
        return info

    @property
    def term_list(self):
        return [self.info]

    @classmethod
    def build_from_info(cls, info):
        terms = []
        for term_info in info['terms']:
            terms.append(SplineTerm.build_from_info(term_info))
        return cls(*terms)

    @property
    def hasconstraint(self):
        constrained = False
        for term in self._terms:
            constrained = constrained or term.hasconstraint
        return constrained

    @property
    def n_coefs(self):
        return np.prod([term.n_coefs for term in self._terms])

    def compile(self, X, verbose=False):
        """
        """
        for term in self._terms:
            term.compile(X, verbose=False)

        if self.by is not None and self.by >= X.shape[1]:
            raise ValueError('by variable requires feature {}, '\
                             'but X has only {} dimensions'\
                             .format(self.by, X.shape[1]))
        return self

    def build_columns(self, X, verbose=False):
        """build a model matrix

        Parameters
        ---------
        X : array-like of shape (n_samples, m_features), default: None
            containing the input dataset


        """
        splines = self._terms[0].build_columns(X, verbose=verbose)
        for term in self._terms[1:]:
            marginal_splines = term.build_columns(X, verbose=verbose)
            splines = tensor_product(splines, marginal_splines)

        if self.by is not None:
            splines *= X[:, self.by][:, np.newaxis]

        return sp.sparse.csc_matrix(splines)

    def build_penalties(self):
        P = np.zeros((self.n_coefs, self.n_coefs))
        for i in range(len(self._terms)):
            P += self._build_marginal_penalties(i)

        return sp.sparse.csc_matrix(P)

    def _build_marginal_penalties(self, i):
        for j, term in enumerate(self._terms):
            # make appropriate marginal penalty
            if j == i:
                P = term.build_penalties()
            else:
                P = sp.sparse.eye(term.n_coefs)

            # compose with other dimensions
            if j == 0:
                P_total = P
            else:
                P_total = sp.sparse.kron(P_total, P)

        return P_total



class TermList(Core):
    def __init__(self, *terms, **kwargs):
        super(TermList, self).__init__()
        # default verbose value, for python 2 compatibility
        self.verbose = kwargs.pop('verbose', False)

        if bool(kwargs):
            raise ValueError("Unexpected keyword argument {}".format(kwargs.keys()))

        def deduplicate(term, term_list, uniques_dict):
            key = str(sorted(term.info.items()))
            if not uniques_dict[key]:
                uniques_dict[key] = True
                term_list.append(term)
            else:
                if self.verbose:
                    warnings.warn('skipping duplicate term: {}'.format(repr(term)))
            return term_list

        # process terms
        uniques = defaultdict(bool)
        term_list = []
        for term in terms:
            if isinstance(term, Term):
                term_list = deduplicate(term, term_list, uniques)
            elif isinstance(term, TermList):
                for term_ in term.term_list:
                    term_list = deduplicate(term_, term_list, uniques)
            else:
                raise ValueError('terms must be instances of Term or TermList, '\
                                 'but found term: {}'.format(term))

        self.term_list = term_list

    def __repr__(self):
        return ' + '.join(repr(term) for term in self)

    def __len__(self):
        return len(self.term_list)

    def __getitem__(self, i):
        return self.term_list[i]

    def __radd__(self, other):
        return TermList(other, self)

    def __add__(self, other):
        return TermList(self, other)

    def __mul__(self, other):
        raise NotImplementedError()

    @property
    def info(self):
        return [term.info for term in self.term_list]

    @classmethod
    def build_from_info(cls, info):
        info = deepcopy(info)
        terms = []
        for term_info in info:
            if 'term_type' in term_info:
                cls_ = TERMS[term_info.pop('term_type')]
            else:
                cls_ = Term
            terms.append(cls_.build_from_info(term_info))
        return cls(*terms)

    def compile(self, X, verbose=False):
        for term in self.term_list:
            term.compile(X, verbose=verbose)

        # now remove duplicate intercepts
        n_intercepts = 0
        for term in self.term_list:
            if term.isintercept:
                n_intercepts += 1
        return self

    def pop(self, i):
        if i >= len(self.term_list):
            raise ValueError('requested pop {}th term, but found only {} terms'\
                            .format(i, len(self.term_list)))

        return TermList(*self.term_list[:i]) + TermList(*self.term_list[i+1:])

    @property
    def hasconstraint(self):
        constrained = False
        for term in self.term_list:
            constrained = constrained or term.hasconstraint
        return constrained

    @property
    def n_coefs(self):
        return sum([term.n_coefs for term in self.term_list])

    def get_coef_indices(self, i=-1):
        if i == -1:
            return list(range(self.n_coefs))

        if i >= len(self.term_list):
            raise ValueError('requested {}th term, but found only {} terms'\
                            .format(i, len(self.term_list)))

        start = 0
        for term in self.term_list[:i]:
            start += term.n_coefs
        stop = start + self.term_list[i].n_coefs
        return list(range(start, stop))

    def build_columns(self, X, term=-1, verbose=False):
        if term == -1:
            term = range(len(self.term_list))
        term = list(np.atleast_1d(term))

        columns = []
        for term_id in term:
            columns.append(self.term_list[term_id].build_columns(X, verbose=verbose))
        return sp.sparse.hstack(columns, format='csc')

    def build_penalties(self):
        P = []
        for term in self.term_list:
            P.append(term.build_penalties())
        return sp.sparse.block_diag(P)

    def build_constraints(self, coefs):
        pass

# Minimal representations
def l(*args, **kwargs):
    return LinearTerm(*args, **kwargs)

def s(*args, **kwargs):
    return SplineTerm(*args, **kwargs)

def f(*args, **kwargs):
    return FactorTerm(*args, **kwargs)

def te(*args, **kwargs):
    return TensorTerm(*args, **kwargs)

# def minimize_repr(term, args, kwargs, minimal_name):
#     unspecified = set(term._get_defaults()) - set(kwargs) - set(['feature'])
#     term._exclude += list(unspecified)
#     term._args = list(args)#[1:]
#     term._minimal_name = minimal_name
#     return term


TERMS = {'term' : Term,
         'intercept_term' : Intercept,
         'linear_term': LinearTerm,
         'spline_term': SplineTerm,
         'factor_term': FactorTerm,
         'tensor_term': TensorTerm,
}

MINIMAL_TERMS = {'l': l,
                 's': s,
                 'te': te,
}