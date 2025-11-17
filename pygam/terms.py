"""Link functions."""

import warnings
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import defaultdict
from copy import deepcopy

import numpy as np
import scipy as sp

from pygam.core import Core, nice_repr
from pygam.penalties import CONSTRAINTS, PENALTIES
from pygam.utils import (
    b_spline_basis,
    check_param,
    flatten,
    gen_edge_knots,
    isiterable,
    tensor_product,
)


class Term(Core):
    __metaclass__ = ABCMeta

    def __init__(
        self,
        feature,
        lam=0.6,
        dtype="numerical",
        fit_linear=False,
        fit_splines=True,
        penalties="auto",
        constraints=None,
        verbose=False,
    ):
        """Creates an instance of a Term.

        Parameters
        ----------
        feature : int
            Index of the feature to use for the feature function.

        lam :  float or iterable of floats
            Strength of smoothing penalty. Must be a positive float.
            Larger values enforce stronger smoothing.

            If single value is passed, it will be repeated for every penalty.

            If iterable is passed, the length of `lam` must be equal to the
            length of `penalties`

        penalties : {'auto', 'derivative', 'l2', None} or callable or iterable
            Type of smoothing penalty to apply to the term.

            If an iterable is used, multiple penalties are applied to the term.
            The length of the iterable must match the length of `lam`.

            If 'auto', then 2nd derivative smoothing for 'numerical' dtypes,
            and L2/ridge smoothing for 'categorical' dtypes.

            Custom penalties can be passed as a callable.

        constraints : {None, 'convex', 'concave', 'monotonic_inc', 'monotonic_dec'}
            or callable or iterable

            Type of constraint to apply to the term.

            If an iterable is used, multiple penalties are applied to the term.

        dtype : {'numerical', 'categorical'}
            String describing the data-type of the feature.

        fit_linear : bool
            whether to fit a linear model of the feature

        fit_splines : bool
            whether to fit spliens to the feature

        Attributes
        ----------
        n_coefs : int
            Number of coefficients contributed by the term to the model

        istensor : bool
            whether the term is a tensor product of sub-terms

        isintercept : bool
            whether the term is an intercept

        hasconstraint : bool
            whether the term has any constraints

        info : dict
            contains dict with the sufficient information to duplicate the term
        """
        self.feature = feature

        self.lam = lam
        self.dtype = dtype
        self.fit_linear = fit_linear
        self.fit_splines = fit_splines
        self.penalties = penalties
        self.constraints = constraints
        self.verbose = verbose

        if not (hasattr(self, "_name")):
            self._name = "term"

        super(Term, self).__init__(name=self._name)
        self._validate_arguments()

    def __len__(self):
        return 1

    def __eq__(self, other):
        if isinstance(other, Term):
            return self.info == other.info
        return False

    def __radd__(self, other):
        return TermList(other, self)

    def __add__(self, other):
        return TermList(self, other)

    def __mul__(self, other):
        raise NotImplementedError()

    def __repr__(self):
        if hasattr(self, "_minimal_name"):
            name = self._minimal_name
        else:
            name = self.__class__.__name__

        features = [] if self.feature is None else self.feature
        features = np.atleast_1d(features).tolist()
        return nice_repr(
            name,
            {},
            line_width=self._line_width,
            line_offset=self._line_offset,
            decimals=4,
            args=features,
        )

    def _validate_arguments(self):
        """Method to sanitize model parameters.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # dtype
        if self.dtype not in ["numerical", "categorical"]:
            raise ValueError(
                "dtype must be in ['numerical','categorical'], "
                f"but found dtype = {self.dtype}"
            )

        # fit_linear XOR fit_splines
        if self.fit_linear == self.fit_splines:
            raise ValueError(
                "term must have fit_linear XOR fit_splines, but found: "
                f"fit_linear= {self.fit_linear}, fit_splines={self.fit_splines}"
            )

        # penalties
        if not isiterable(self.penalties):
            self.penalties = [self.penalties]

        for i, p in enumerate(self.penalties):
            if not (hasattr(p, "__call__") or (p in PENALTIES) or (p is None)):
                raise ValueError(
                    "penalties must be callable or in "
                    f"{list(PENALTIES.keys())}, but found {p} for {i}th penalty"
                )

        # check lams and distribute to penalties
        if not isiterable(self.lam):
            self.lam = [self.lam]

        for lam in self.lam:
            check_param(lam, param_name="lam", dtype="float", constraint=">= 0")

        if len(self.lam) == 1:
            self.lam = self.lam * len(self.penalties)

        if len(self.lam) != len(self.penalties):
            raise ValueError(
                f"expected 1 lam per penalty, but found lam = {self.lam}, penalties = {self.penalties}"
            )

        # constraints
        if not isiterable(self.constraints):
            self.constraints = [self.constraints]

        for i, c in enumerate(self.constraints):
            if not (hasattr(c, "__call__") or (c in CONSTRAINTS) or (c is None)):
                raise ValueError(
                    "constraints must be callable or in "
                    f"{list(CONSTRAINTS.keys())}, but found {c} for {i}th constraint"
                )

        return self

    @property
    def istensor(self):
        return isinstance(self, TensorTerm)

    @property
    def isintercept(self):
        return isinstance(self, Intercept)

    @property
    def info(self):
        """Get information about this term.

        Parameters
        ----------

        Returns
        -------
        dict containing information to duplicate this term
        """
        info = self.get_params()
        info.update({"term_type": self._name})
        return info

    @classmethod
    def build_from_info(cls, info):
        """Build a Term instance from a dict.

        Parameters
        ----------
        cls : class

        info : dict
            contains all information needed to build the term

        Return
        ------
        Term instance
        """
        info = deepcopy(info)
        if "term_type" in info:
            cls_ = TERMS[info.pop("term_type")]

            if issubclass(cls_, MetaTermMixin):
                return cls_.build_from_info(info)
        else:
            cls_ = cls
        return cls_(**info)

    @property
    def hasconstraint(self):
        """bool, whether the term has any constraints."""
        return np.not_equal(np.atleast_1d(self.constraints), None).any()

    @property
    @abstractproperty
    def n_coefs(self):
        """Number of coefficients contributed by the term to the model."""
        pass

    @abstractmethod
    def compile(self, X, verbose=False):
        """Method to validate and prepare data-dependent parameters.

        Parameters
        ----------
        X : array-like
            Input dataset

        verbose : bool
            whether to show warnings

        Returns
        -------
        None
        """
        return self

    @abstractmethod
    def build_columns(self, X, verbose=False):
        """Construct the model matrix columns for the term.

        Parameters
        ----------
        X : array-like
            Input dataset with n rows

        verbose : bool
            whether to show warnings

        Returns
        -------
        scipy sparse array with n rows
        """
        pass

    def build_penalties(self, verbose=False):
        """
        Builds the GAM block-diagonal penalty matrix in quadratic form
        out of penalty matrices specified for each feature.

        each feature penalty matrix is multiplied by a lambda for that feature.

        so for m features:
        P = block_diag[lam0 * P0, lam1 * P1, lam2 * P2, ... , lamm * Pm]


        Parameters
        ----------
        None

        Returns
        -------
        P : sparse CSC matrix containing the model penalties in quadratic form
        """
        if self.isintercept:
            return np.array([[0.0]])

        Ps = []
        for penalty, lam in zip(self.penalties, self.lam):
            if penalty == "auto":
                if self.dtype == "numerical":
                    if self._name == "spline_term":
                        if self.basis in ["cp"]:
                            penalty = "periodic"
                        else:
                            penalty = "derivative"
                    else:
                        penalty = "l2"
                if self.dtype == "categorical":
                    penalty = "l2"
            if penalty is None:
                penalty = "none"
            if penalty in PENALTIES:
                penalty = PENALTIES[penalty]

            P = penalty(self.n_coefs, coef=None)  # penalties dont need coef
            Ps.append(np.multiply(P, lam))
        return np.sum(Ps)

    def build_constraints(self, coef, constraint_lam, constraint_l2):
        """
        Builds the GAM block-diagonal constraint matrix in quadratic form
        out of constraint matrices specified for each feature.

        behaves like a penalty, but with a very large lambda value, ie 1e6.

        Parameters
        ----------
        coefs : array-like containing the coefficients of a term

        constraint_lam : float,
            penalty to impose on the constraint.

            typically this is a very large number.

        constraint_l2 : float,
            loading to improve the numerical conditioning of the constraint
            matrix.

            typically this is a very small number.

        Returns
        -------
        C : sparse CSC matrix containing the model constraints in quadratic form
        """
        if self.isintercept:
            return np.array([[0.0]])

        Cs = []
        for constraint in self.constraints:
            if constraint is None:
                constraint = "none"
            if constraint in CONSTRAINTS:
                constraint = CONSTRAINTS[constraint]

            C = constraint(self.n_coefs, coef) * constraint_lam
            Cs.append(C)

        Cs = np.sum(Cs)

        # improve condition
        if Cs.nnz > 0:
            Cs += sp.sparse.diags(constraint_l2 * np.ones(Cs.shape[0]))

        return Cs


class Intercept(Term):
    def __init__(self, verbose=False):
        """Creates an instance of an Intercept term.

        Parameters
        ----------

        Attributes
        ----------
        n_coefs : int
            Number of coefficients contributed by the term to the model

        istensor : bool
            whether the term is a tensor product of sub-terms

        isintercept : bool
            whether the term is an intercept

        hasconstraint : bool
            whether the term has any constraints

        info : dict
            contains dict with the sufficient information to duplicate the term
        """
        self._name = "intercept_term"
        self._minimal_name = "intercept"

        super(Intercept, self).__init__(
            feature=None,
            fit_linear=False,
            fit_splines=False,
            lam=None,
            penalties=None,
            constraints=None,
            verbose=verbose,
        )

        self._exclude += [
            "fit_splines",
            "fit_linear",
            "lam",
            "penalties",
            "constraints",
            "feature",
            "dtype",
        ]
        self._args = []

    def __repr__(self):
        return self._minimal_name

    def _validate_arguments(self):
        """Method to sanitize model parameters.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return self

    @property
    def n_coefs(self):
        """Number of coefficients contributed by the term to the model."""
        return 1

    def compile(self, X, verbose=False):
        """Method to validate and prepare data-dependent parameters.

        Parameters
        ----------
        X : array-like
            Input dataset

        verbose : bool
            whether to show warnings

        Returns
        -------
        None
        """
        return self

    def build_columns(self, X, verbose=False):
        """Construct the model matrix columns for the term.

        Parameters
        ----------
        X : array-like
            Input dataset with n rows

        verbose : bool
            whether to show warnings

        Returns
        -------
        scipy sparse array with n rows
        """
        return sp.sparse.csc_matrix(np.ones((len(X), 1)))


class LinearTerm(Term):
    def __init__(self, feature, lam=0.6, penalties="auto", verbose=False):
        """Creates an instance of a LinearTerm.

        Parameters
        ----------
        feature : int
            Index of the feature to use for the feature function.

        lam :  float or iterable of floats
            Strength of smoothing penalty. Must be a positive float.
            Larger values enforce stronger smoothing.

            If single value is passed, it will be repeated for every penalty.

            If iterable is passed, the length of `lam` must be equal to the
            length of `penalties`

        penalties : {'auto', 'derivative', 'l2', None} or callable or iterable
            Type of smoothing penalty to apply to the term.

            If an iterable is used, multiple penalties are applied to the term.
            The length of the iterable must match the length of `lam`.

            If 'auto', then 2nd derivative smoothing for 'numerical' dtypes,
            and L2/ridge smoothing for 'categorical' dtypes.

            Custom penalties can be passed as a callable.

        Attributes
        ----------
        n_coefs : int
            Number of coefficients contributed by the term to the model

        istensor : bool
            whether the term is a tensor product of sub-terms

        isintercept : bool
            whether the term is an intercept

        hasconstraint : bool
            whether the term has any constraints

        info : dict
            contains dict with the sufficient information to duplicate the term
        """
        self._name = "linear_term"
        self._minimal_name = "l"
        super(LinearTerm, self).__init__(
            feature=feature,
            lam=lam,
            penalties=penalties,
            constraints=None,
            dtype="numerical",
            fit_linear=True,
            fit_splines=False,
            verbose=verbose,
        )
        self._exclude += ["fit_splines", "fit_linear", "dtype", "constraints"]

    @property
    def n_coefs(self):
        """Number of coefficients contributed by the term to the model."""
        return 1

    def compile(self, X, verbose=False):
        """Method to validate and prepare data-dependent parameters.

        Parameters
        ----------
        X : array-like
            Input dataset

        verbose : bool
            whether to show warnings

        Returns
        -------
        None
        """
        if self.feature >= X.shape[1]:
            raise ValueError(
                f"term requires feature {self.feature}, but X has only {X.shape[1]} dimensions"
            )

        self.edge_knots_ = gen_edge_knots(
            X[:, self.feature], self.dtype, verbose=verbose
        )
        return self

    def build_columns(self, X, verbose=False):
        """Construct the model matrix columns for the term.

        Parameters
        ----------
        X : array-like
            Input dataset with n rows

        verbose : bool
            whether to show warnings

        Returns
        -------
        scipy sparse array with n rows
        """
        return sp.sparse.csc_matrix(X[:, self.feature][:, np.newaxis])


class SplineTerm(Term):
    _bases = ["ps", "cp"]

    def __init__(
        self,
        feature,
        n_splines=20,
        spline_order=3,
        lam=0.6,
        penalties="auto",
        constraints=None,
        dtype="numerical",
        basis="ps",
        by=None,
        edge_knots=None,
        verbose=False,
    ):
        """Creates an instance of a SplineTerm.

        Parameters
        ----------
        feature : int
            Index of the feature to use for the feature function.

        n_splines : int
            Number of splines to use for the feature function.
            Must be non-negative.

        spline_order : int
            Order of spline to use for the feature function.
            Must be non-negative.

        lam :  float or iterable of floats
            Strength of smoothing penalty. Must be a positive float.
            Larger values enforce stronger smoothing.

            If single value is passed, it will be repeated for every penalty.

            If iterable is passed, the length of `lam` must be equal to the
            length of `penalties`

        penalties : {'auto', 'derivative', 'l2', None} or callable or iterable
            Type of smoothing penalty to apply to the term.

            If an iterable is used, multiple penalties are applied to the term.
            The length of the iterable must match the length of `lam`.

            If 'auto', then 2nd derivative smoothing for 'numerical' dtypes,
            and L2/ridge smoothing for 'categorical' dtypes.

            Custom penalties can be passed as a callable.

        constraints : {None, 'convex', 'concave', 'monotonic_inc', 'monotonic_dec'}
            or callable or iterable

            Type of constraint to apply to the term.

            If an iterable is used, multiple penalties are applied to the term.

        dtype : {'numerical', 'categorical'}
            String describing the data-type of the feature.

        basis : {'ps', 'cp'}
            Type of basis function to use in the term.

            'ps' : p-spline basis

            'cp' : cyclic p-spline basis, useful for building periodic functions.
                   by default, the maximum and minimum of the feature values
                   are used to determine the function's period.

                   to specify a custom period use argument `edge_knots`

        edge_knots : optional, array-like of floats of length 2

            these values specify minimum and maximum domain of the spline function.

            in the case that `spline_basis="cp"`, `edge_knots` determines
            the period of the cyclic function.

            when `edge_knots=None` these values are inferred from the data.

            default: None

        by : int, optional
            Feature to use as a by-variable in the term.

            For example, if `feature` = 2 `by` = 0, then the term will produce:
            x0 * f(x2)

        Attributes
        ----------
        n_coefs : int
            Number of coefficients contributed by the term to the model

        istensor : bool
            whether the term is a tensor product of sub-terms

        isintercept : bool
            whether the term is an intercept

        hasconstraint : bool
            whether the term has any constraints

        info : dict
            contains dict with the sufficient information to duplicate the term
        """
        self.basis = basis
        self.n_splines = n_splines
        self.spline_order = spline_order
        self.by = by
        self._name = "spline_term"
        self._minimal_name = "s"

        if edge_knots is not None:
            self.edge_knots_ = edge_knots

        super(SplineTerm, self).__init__(
            feature=feature,
            lam=lam,
            penalties=penalties,
            constraints=constraints,
            fit_linear=False,
            fit_splines=True,
            dtype=dtype,
            verbose=verbose,
        )

        self._exclude += ["fit_linear", "fit_splines"]

    def _validate_arguments(self):
        """Method to sanitize model parameters.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        super(SplineTerm, self)._validate_arguments()

        if self.basis not in self._bases:
            raise ValueError(
                f"basis must be one of {self._bases}, but found: {self.basis}"
            )

        # n_splines
        self.n_splines = check_param(
            self.n_splines, param_name="n_splines", dtype="int", constraint=">= 0"
        )

        # spline_order
        self.spline_order = check_param(
            self.spline_order, param_name="spline_order", dtype="int", constraint=">= 0"
        )

        # n_splines + spline_order
        if not self.n_splines > self.spline_order:
            raise ValueError(
                "n_splines must be > spline_order. "
                f"found: n_splines = {self.n_splines} and spline_order = {self.spline_order}"
            )

        # by
        if self.by is not None:
            self.by = check_param(
                self.by, param_name="by", dtype="int", constraint=">= 0"
            )

        return self

    @property
    def n_coefs(self):
        """Number of coefficients contributed by the term to the model."""
        return self.n_splines

    def compile(self, X, verbose=False):
        """Method to validate and prepare data-dependent parameters.

        Parameters
        ----------
        X : array-like
            Input dataset

        verbose : bool
            whether to show warnings

        Returns
        -------
        None
        """
        if self.feature >= X.shape[1]:
            raise ValueError(
                f"term requires feature {self.feature}, but X has only {X.shape[1]} dimensions"
            )

        if self.by is not None and self.by >= X.shape[1]:
            raise ValueError(
                f"by variable requires feature {self.by}, but X has only {X.shape[1]} dimensions"
            )

        if not hasattr(self, "edge_knots_"):
            self.edge_knots_ = gen_edge_knots(
                X[:, self.feature], self.dtype, verbose=verbose
            )
        return self

    def build_columns(self, X, verbose=False):
        """Construct the model matrix columns for the term.

        Parameters
        ----------
        X : array-like
            Input dataset with n rows

        verbose : bool
            whether to show warnings

        Returns
        -------
        scipy sparse array with n rows
        """
        X[:, self.feature][:, np.newaxis]

        splines = b_spline_basis(
            X[:, self.feature],
            edge_knots=self.edge_knots_,
            spline_order=self.spline_order,
            n_splines=self.n_splines,
            sparse=True,
            periodic=self.basis in ["cp"],
            verbose=verbose,
        )

        if self.by is not None:
            splines = splines.multiply(X[:, self.by][:, np.newaxis])

        return splines


class FactorTerm(SplineTerm):
    _encodings = ["one-hot", "dummy"]

    def __init__(
        self, feature, lam=0.6, penalties="auto", coding="one-hot", verbose=False
    ):
        """Creates an instance of a FactorTerm.

        Parameters
        ----------
        feature : int
            Index of the feature to use for the feature function.

        lam :  float or iterable of floats
            Strength of smoothing penalty. Must be a positive float.
            Larger values enforce stronger smoothing.

            If single value is passed, it will be repeated for every penalty.

            If iterable is passed, the length of `lam` must be equal to the
            length of `penalties`

        penalties : {'auto', 'derivative', 'l2', None} or callable or iterable
            Type of smoothing penalty to apply to the term.

            If an iterable is used, multiple penalties are applied to the term.
            The length of the iterable must match the length of `lam`.

            If 'auto', then 2nd derivative smoothing for 'numerical' dtypes,
            and L2/ridge smoothing for 'categorical' dtypes.

            Custom penalties can be passed as a callable.

        coding : {'one-hot'} type of contrast encoding to use.
            currently, only 'one-hot' encoding has been developed.
            this means that we fit one coefficient per category.

        Attributes
        ----------
        n_coefs : int
            Number of coefficients contributed by the term to the model

        istensor : bool
            whether the term is a tensor product of sub-terms

        isintercept : bool
            whether the term is an intercept

        hasconstraint : bool
            whether the term has any constraints

        info : dict
            contains dict with the sufficient information to duplicate the term
        """
        self.coding = coding
        super(FactorTerm, self).__init__(
            feature=feature,
            lam=lam,
            dtype="categorical",
            spline_order=0,
            penalties=penalties,
            by=None,
            constraints=None,
            verbose=verbose,
        )
        self._name = "factor_term"
        self._minimal_name = "f"
        self._exclude += [
            "dtype",
            "spline_order",
            "by",
            "n_splines",
            "basis",
            "constraints",
        ]

    def _validate_arguments(self):
        """Method to sanitize model parameters.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        super(FactorTerm, self)._validate_arguments()
        if self.coding not in self._encodings:
            raise ValueError(
                f"coding must be one of {self._encodings}, but found: {self.coding}"
            )

        return self

    def compile(self, X, verbose=False):
        """Method to validate and prepare data-dependent parameters.

        Parameters
        ----------
        X : array-like
            Input dataset

        verbose : bool
            whether to show warnings

        Returns
        -------
        None
        """
        super(FactorTerm, self).compile(X)

        self.n_splines = len(np.unique(X[:, self.feature]))
        self.edge_knots_ = gen_edge_knots(
            X[:, self.feature], self.dtype, verbose=verbose
        )
        return self

    def build_columns(self, X, verbose=False):
        """Construct the model matrix columns for the term.

        Parameters
        ----------
        X : array-like
            Input dataset with n rows

        verbose : bool
            whether to show warnings

        Returns
        -------
        scipy sparse array with n rows
        """
        columns = super(FactorTerm, self).build_columns(X, verbose=verbose)
        if self.coding == "dummy":
            columns = columns[:, 1:]

        return columns

    @property
    def n_coefs(self):
        """Number of coefficients contributed by the term to the model."""
        return self.n_splines - 1 * (self.coding in ["dummy"])


class MetaTermMixin:
    _plural = [
        "feature",
        "dtype",
        "fit_linear",
        "fit_splines",
        "lam",
        "n_splines",
        "spline_order",
        "constraints",
        "penalties",
        "basis",
        "edge_knots_",
    ]
    _term_location = "_terms"

    def _super_get(self, name):
        return super(MetaTermMixin, self).__getattribute__(name)

    def _super_has(self, name):
        try:
            self._super_get(name)
            return True
        except AttributeError:
            return False

    def _has_terms(self):
        """bool, whether the instance has any sub-terms."""
        loc = self._super_get("_term_location")
        return (
            self._super_has(loc)
            and isiterable(self._super_get(loc))
            and len(self._super_get(loc)) > 0
            and all([isinstance(term, Term) for term in self._super_get(loc)])
        )

    def _get_terms(self):
        """Get the terms in the instance.

        Parameters
        ----------
        None

        Returns
        -------
        list containing terms
        """
        if self._has_terms():
            return getattr(self, self._term_location)

    def __setattr__(self, name, value):
        if self._has_terms() and name in self._super_get("_plural"):
            # get the total number of arguments
            size = np.atleast_1d(flatten(getattr(self, name))).size

            # check shapes
            if isiterable(value):
                value = flatten(value)
                if len(value) != size:
                    raise ValueError(
                        f"Expected {name} to have length {size}, but found {name} = {value}"
                    )
            else:
                value = [value] * size

            # now set each term's sequence of arguments
            for term in self._get_terms()[::-1]:
                # skip intercept
                if term.isintercept:
                    continue

                # how many values does this term get?
                n = np.atleast_1d(getattr(term, name)).size

                # get the next n values and set them on this term
                vals = [value.pop() for _ in range(n)][::-1]
                setattr(term, name, vals[0] if n == 1 else vals)

                term._validate_arguments()

            return
        super(MetaTermMixin, self).__setattr__(name, value)

    def __getattr__(self, name):
        if self._has_terms() and name in self._super_get("_plural"):
            # collect value from each term
            values = []
            for term in self._get_terms():
                # skip the intercept
                if term.isintercept:
                    continue

                values.append(getattr(term, name, None))
            return values

        return self._super_get(name)


class TensorTerm(SplineTerm, MetaTermMixin):
    _N_SPLINES = 10  # default num splines

    def __init__(self, *args, **kwargs):
        """Creates an instance of a TensorTerm.

        This is useful for creating interactions between features, or other terms.

        Parameters
        ----------
        *args : marginal Terms to combine into a tensor product

        feature : list of integers
            Indices of the features to use for the marginal terms.

        n_splines : list of integers
            Number of splines to use for each marginal term.
            Must be of same length as `feature`.

        spline_order : list of integers
            Order of spline to use for the feature function.
            Must be of same length as `feature`.

        lam :  float or iterable of floats
            Strength of smoothing penalty. Must be a positive float.
            Larger values enforce stronger smoothing.

            If single value is passed, it will be repeated for every penalty.

            If iterable is passed, the length of `lam` must be equal to the
            length of `penalties`

        penalties : {'auto', 'derivative', 'l2', None} or callable or iterable
            Type of smoothing penalty to apply to the term.

            If an iterable is used, multiple penalties are applied to the term.
            The length of the iterable must match the length of `lam`.

            If 'auto', then 2nd derivative smoothing for 'numerical' dtypes,
            and L2/ridge smoothing for 'categorical' dtypes.

            Custom penalties can be passed as a callable.

        constraints : {None, 'convex', 'concave', 'monotonic_inc', 'monotonic_dec'}
            or callable or iterable

            Type of constraint to apply to the term.

            If an iterable is used, multiple penalties are applied to the term.

        dtype : list of {'numerical', 'categorical'}
            String describing the data-type of the feature.

            Must be of same length as `feature`.

        basis : list of {'ps'}
            Type of basis function to use in the term.

            'ps' : p-spline basis

            NotImplemented:
            'cp' : cyclic p-spline basis

            Must be of same length as `feature`.

        by : int, optional
            Feature to use as a by-variable in the term.

            For example, if `feature` = [1, 2] `by` = 0, then the term will produce:
            x0 * te(x1, x2)

        Attributes
        ----------
        n_coefs : int
            Number of coefficients contributed by the term to the model

        istensor : bool
            whether the term is a tensor product of sub-terms

        isintercept : bool
            whether the term is an intercept

        hasconstraint : bool
            whether the term has any constraints

        info : dict
            contains dict with the sufficient information to duplicate the term
        """
        self.verbose = kwargs.pop("verbose", False)
        by = kwargs.pop("by", None)
        terms = self._parse_terms(args, **kwargs)

        feature = [term.feature for term in terms]
        super(TensorTerm, self).__init__(feature, by=by, verbose=self.verbose)

        self._name = "tensor_term"
        self._minimal_name = "te"

        self._exclude = [
            "feature",
            "dtype",
            "fit_linear",
            "fit_splines",
            "lam",
            "n_splines",
            "spline_order",
            "constraints",
            "penalties",
            "basis",
        ]
        for param in self._exclude:
            delattr(self, param)

        self._terms = terms

    def _parse_terms(self, args, **kwargs):
        m = len(args)
        if m < 2:
            raise ValueError("TensorTerm requires at least 2 marginal terms")

        for k, v in kwargs.items():
            if isiterable(v):
                if len(v) != m:
                    raise ValueError(
                        f"Expected {k} to have length {m}, but found {k} = {v}"
                    )
            else:
                kwargs[k] = [v] * m

        terms = []
        for i, arg in enumerate(np.atleast_1d(args)):
            if isinstance(arg, TensorTerm):
                raise ValueError(
                    "TensorTerm does not accept other TensorTerms. "
                    "Please build a flat TensorTerm instead of a nested one."
                )

            if isinstance(arg, Term):
                if self.verbose and kwargs:
                    warnings.warn(
                        "kwargs are skipped when Term instances are passed to TensorTerm constructor"  # noqa: E501
                    )
                terms.append(arg)
                continue

            kwargs_ = {"n_splines": self._N_SPLINES}
            kwargs_.update({k: v[i] for k, v in kwargs.items()})

            terms.append(SplineTerm(arg, **kwargs_))

        return terms

    def __len__(self):
        return len(self._terms)

    def __getitem__(self, i):
        return self._terms[i]

    def _validate_arguments(self):
        """Method to sanitize model parameters.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self._has_terms():
            [term._validate_arguments() for term in self._terms]
        else:
            super(TensorTerm, self)._validate_arguments()

        return self

    @property
    def info(self):
        """Get information about this term.

        Parameters
        ----------

        Returns
        -------
        dict containing information to duplicate this term
        """
        info = super(TensorTerm, self).info
        info.update({"terms": [term.info for term in self._terms]})
        return info

    @classmethod
    def build_from_info(cls, info):
        """Build a TensorTerm instance from a dict.

        Parameters
        ----------
        cls : class

        info : dict
            contains all information needed to build the term

        Return
        ------
        TensorTerm instance
        """
        terms = []
        for term_info in info["terms"]:
            terms.append(SplineTerm.build_from_info(term_info))
        return cls(*terms)

    @property
    def hasconstraint(self):
        """bool, whether the term has any constraints."""
        constrained = False
        for term in self._terms:
            constrained = constrained or term.hasconstraint
        return constrained

    @property
    def n_coefs(self):
        """Number of coefficients contributed by the term to the model."""
        return np.prod([term.n_coefs for term in self._terms])

    def compile(self, X, verbose=False):
        """Method to validate and prepare data-dependent parameters.

        Parameters
        ----------
        X : array-like
            Input dataset

        verbose : bool
            whether to show warnings

        Returns
        -------
        None
        """
        for term in self._terms:
            term.compile(X, verbose=False)

        if self.by is not None and self.by >= X.shape[1]:
            raise ValueError(
                f"by variable requires feature {self.by}, but X has only {X.shape[1]} dimensions"
            )
        return self

    def build_columns(self, X, verbose=False):
        """Construct the model matrix columns for the term.

        Parameters
        ----------
        X : array-like
            Input dataset with n rows

        verbose : bool
            whether to show warnings

        Returns
        -------
        scipy sparse array with n rows
        """
        splines = self._terms[0].build_columns(X, verbose=verbose)
        for term in self._terms[1:]:
            marginal_splines = term.build_columns(X, verbose=verbose)
            splines = tensor_product(splines, marginal_splines)

        if self.by is not None:
            splines *= X[:, self.by][:, np.newaxis]

        return sp.sparse.csc_matrix(splines)

    def build_penalties(self):
        """
        Builds the GAM block-diagonal penalty matrix in quadratic form
        out of penalty matrices specified for each feature.

        each feature penalty matrix is multiplied by a lambda for that feature.

        so for m features:
        P = block_diag[lam0 * P0, lam1 * P1, lam2 * P2, ... , lamm * Pm]

        Parameters
        ----------
        None

        Returns
        -------
        P : sparse CSC matrix containing the model penalties in quadratic form
        """
        P = sp.sparse.csc_matrix(np.zeros((self.n_coefs, self.n_coefs)))
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

    def build_constraints(self, coef, constraint_lam, constraint_l2):
        """
        Builds the GAM block-diagonal constraint matrix in quadratic form
        out of constraint matrices specified for each feature.

        Parameters
        ----------
        coefs : array-like containing the coefficients of a term

        constraint_lam : float,
            penalty to impose on the constraint.

            typically this is a very large number.

        constraint_l2 : float,
            loading to improve the numerical conditioning of the constraint
            matrix.

            typically this is a very small number.

        Returns
        -------
        C : sparse CSC matrix containing the model constraints in quadratic form
        """
        C = sp.sparse.csc_matrix(np.zeros((self.n_coefs, self.n_coefs)))
        for i in range(len(self._terms)):
            C += self._build_marginal_constraints(
                i, coef, constraint_lam, constraint_l2
            )

        return sp.sparse.csc_matrix(C)

    def _build_marginal_constraints(self, i, coef, constraint_lam, constraint_l2):
        """Builds a constraint matrix for a marginal term in the tensor term.

        takes a tensor's coef vector, and slices it into pieces corresponding
        to term i, then builds a constraint matrix for each piece of the coef vector,
        and assembles them into a composite constraint matrix

        Parameters
        ----------
        i : int,
            index of the marginal term for which to build a constraint matrix

        coefs : array-like containing the coefficients of the tensor term

        constraint_lam : float,
            penalty to impose on the constraint.

            typically this is a very large number.

        constraint_l2 : float,
            loading to improve the numerical conditioning of the constraint
            matrix.

            typically this is a very small number.

        Returns
        -------
        C : sparse CSC matrix containing the model constraints in quadratic form
        """
        composite_C = np.zeros((len(coef), len(coef)))

        for slice_ in self._iterate_marginal_coef_slices(i):
            # get the slice of coefficient vector
            coef_slice = coef[slice_]

            # build the constraint matrix for that slice
            slice_C = self._terms[i].build_constraints(
                coef_slice, constraint_lam, constraint_l2
            )

            # now enter it into the composite
            composite_C[tuple(np.meshgrid(slice_, slice_))] = slice_C.toarray()

        return sp.sparse.csc_matrix(composite_C)

    def _iterate_marginal_coef_slices(self, i):
        """Iterator of indices into tensor's coef vector for marginal term i's coefs.

        takes a tensor_term and returns an iterator of indices
        that chop up the tensor's coef vector into slices belonging to term i

        Parameters
        ----------
        i : int,
            index of marginal term

        Yields
        ------
        np.ndarray of ints
        """
        dims = [term_.n_coefs for term_ in self]

        # make all linear indices
        idxs = np.arange(np.prod(dims))

        # reshape indices to a Nd matrix
        idxs = idxs.reshape(dims)

        # reshape to a 2d matrix, where we can loop over rows
        idxs = np.moveaxis(idxs, i, 0).reshape(
            idxs.shape[i], int(idxs.size / idxs.shape[i])
        )

        # loop over rows
        for slice_ in idxs.T:
            yield slice_


class TermList(Core, MetaTermMixin):
    _terms = []

    def __init__(self, *terms, **kwargs):
        """Creates an instance of a TermList.

        If duplicate terms are supplied, only the first instance will be kept.

        Parameters
        ----------
        *terms : list of terms to

        verbose : bool
            whether to show warnings

        Attributes
        ----------
        n_coefs : int
            Total number of coefficients in the model

        hasconstraint : bool
            whether the model has any constraints

        info : dict
            contains dict with the sufficient information to duplicate the term list
        """
        super(TermList, self).__init__()
        self.verbose = kwargs.pop("verbose", False)

        if bool(kwargs):
            raise ValueError(f"Unexpected keyword argument {kwargs.keys()}")

        def deduplicate(term, term_list, uniques_dict):
            """Adds a term to the term_list only if it is new.

            Parameters
            ----------
            term : Term
                new term in consideration

            term_list : list
                contains all unique terms

            uniques_dict : defaultdict
                keys are term info,
                values are bool: True if the term has been seen already

            Returns
            -------
            term_list : list
                contains `term` if it was unique
            """
            key = str(sorted(term.info.items()))
            if not uniques_dict[key]:
                uniques_dict[key] = True
                term_list.append(term)
            else:
                if self.verbose:
                    warnings.warn(f"skipping duplicate term: {repr(term)}")
            return term_list

        # process terms
        uniques = defaultdict(bool)
        term_list = []
        for term in terms:
            if isinstance(term, Term):
                term_list = deduplicate(term, term_list, uniques)
            elif isinstance(term, TermList):
                for term_ in term._terms:
                    term_list = deduplicate(term_, term_list, uniques)
            else:
                raise ValueError(
                    "terms must be instances of Term or TermList, "
                    f"but found term: {term}"
                )

        self._terms = self._terms + term_list
        self._exclude = [
            "feature",
            "dtype",
            "fit_linear",
            "fit_splines",
            "lam",
            "n_splines",
            "spline_order",
            "constraints",
            "penalties",
            "basis",
        ]
        self.verbose = any([term.verbose for term in self._terms]) or self.verbose

    def __eq__(self, other):
        if isinstance(other, TermList):
            return self.info == other.info
        return False

    def __repr__(self):
        return " + ".join(repr(term) for term in self)

    def __len__(self):
        return len(self._terms)

    def __getitem__(self, i):
        return self._terms[i]

    def __radd__(self, other):
        return TermList(other, self)

    def __add__(self, other):
        return TermList(self, other)

    def __mul__(self, other):
        raise NotImplementedError()

    def _validate_arguments(self):
        """Method to sanitize model parameters.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self._has_terms():
            [term._validate_arguments() for term in self._terms]
        return self

    @property
    def info(self):
        """Get information about the terms in the term list.

        Parameters
        ----------

        Returns
        -------
        dict containing information to duplicate the term list
        """
        info = {"term_type": "term_list", "verbose": self.verbose}
        info.update({"terms": [term.info for term in self._terms]})
        return info

    @classmethod
    def build_from_info(cls, info):
        """Build a TermList instance from a dict.

        Parameters
        ----------
        cls : class

        info : dict
            contains all information needed to build the term

        Return
        ------
        TermList instance
        """
        info = deepcopy(info)
        terms = []
        for term_info in info["terms"]:
            terms.append(Term.build_from_info(term_info))
        return cls(*terms)

    def compile(self, X, verbose=False):
        """Method to validate and prepare data-dependent parameters.

        Parameters
        ----------
        X : array-like
            Input dataset

        verbose : bool
            whether to show warnings

        Returns
        -------
        None
        """
        for term in self._terms:
            term.compile(X, verbose=verbose)

        # now remove duplicate intercepts
        n_intercepts = 0
        for term in self._terms:
            if term.isintercept:
                n_intercepts += 1
        return self

    def pop(self, i=None):
        """Remove the ith term from the term list.

        Parameters
        ----------
        i : int, optional
            term to remove from term list

            by default the last term is popped.

        Returns
        -------
        term : Term
        """
        if i is None:
            i = len(self) - 1

        if i >= len(self._terms) or i < 0:
            raise ValueError(
                f"requested pop {i}th term, but found only {len(self._terms)} terms"
            )

        term = self._terms[i]
        self._terms = self._terms[:i] + self._terms[i + 1 :]
        return term

    @property
    def hasconstraint(self):
        """bool, whether the term has any constraints."""
        constrained = False
        for term in self._terms:
            constrained = constrained or term.hasconstraint
        return constrained

    @property
    def n_coefs(self):
        """Total number of coefficients contributed by the terms in the model."""
        return sum([term.n_coefs for term in self._terms])

    def get_coef_indices(self, i=-1):
        """Get the indices for the coefficients of a term in the term list.

        Parameters
        ----------
        i : int
            by default `int=-1`, meaning that coefficient indices are returned
            for all terms in the term list

        Returns
        -------
        list of integers
        """
        if i == -1:
            return list(range(self.n_coefs))

        if i >= len(self._terms):
            raise ValueError(
                f"requested {i}th term, but found only {len(self._terms)} terms"
            )

        start = 0
        for term in self._terms[:i]:
            start += term.n_coefs
        stop = start + self._terms[i].n_coefs
        return list(range(start, stop))

    def build_columns(self, X, term=-1, verbose=False):
        """Construct the model matrix columns for the term.

        Parameters
        ----------
        X : array-like
            Input dataset with n rows

        verbose : bool
            whether to show warnings

        Returns
        -------
        scipy sparse array with n rows
        """
        if term == -1:
            term = range(len(self._terms))
        term = list(np.atleast_1d(term))

        columns = []
        for term_id in term:
            columns.append(self._terms[term_id].build_columns(X, verbose=verbose))
        return sp.sparse.hstack(columns, format="csc")

    def build_penalties(self):
        """
        Builds the GAM block-diagonal penalty matrix in quadratic form
        out of penalty matrices specified for each feature.

        each feature penalty matrix is multiplied by a lambda for that feature.

        so for m features:
        P = block_diag[lam0 * P0, lam1 * P1, lam2 * P2, ... , lamm * Pm]


        Parameters
        ----------
        None

        Returns
        -------
        P : sparse CSC matrix containing the model penalties in quadratic form
        """
        P = []
        for term in self._terms:
            P.append(term.build_penalties())
        return sp.sparse.block_diag(P)

    def build_constraints(self, coefs, constraint_lam, constraint_l2):
        """
        Builds the GAM block-diagonal constraint matrix in quadratic form
        out of constraint matrices specified for each feature.

        behaves like a penalty, but with a very large lambda value, ie 1e6.

        Parameters
        ----------
        coefs : array-like containing the coefficients of a term

        constraint_lam : float,
            penalty to impose on the constraint.

            typically this is a very large number.

        constraint_l2 : float,
            loading to improve the numerical conditioning of the constraint
            matrix.

            typically this is a very small number.

        Returns
        -------
        C : sparse CSC matrix containing the model constraints in quadratic form
        """
        C = []
        for i, term in enumerate(self._terms):
            idxs = self.get_coef_indices(i=i)
            C.append(term.build_constraints(coefs[idxs], constraint_lam, constraint_l2))
        return sp.sparse.block_diag(C)


# Minimal representations
def l(feature, lam=0.6, penalties="auto", verbose=False):  # noqa: E743
    """

    See Also
    --------
    LinearTerm : for developer details
    """
    return LinearTerm(feature=feature, lam=lam, penalties=penalties, verbose=verbose)


def s(
    feature,
    n_splines=20,
    spline_order=3,
    lam=0.6,
    penalties="auto",
    constraints=None,
    dtype="numerical",
    basis="ps",
    by=None,
    edge_knots=None,
    verbose=False,
):
    """

    See Also
    --------
    SplineTerm : for developer details
    """
    return SplineTerm(
        feature=feature,
        n_splines=n_splines,
        spline_order=spline_order,
        lam=lam,
        penalties=penalties,
        constraints=constraints,
        dtype=dtype,
        basis=basis,
        by=by,
        edge_knots=edge_knots,
        verbose=verbose,
    )


def f(feature, lam=0.6, penalties="auto", coding="one-hot", verbose=False):
    """

    See Also
    --------
    FactorTerm : for developer details
    """
    return FactorTerm(
        feature=feature, lam=lam, penalties=penalties, coding=coding, verbose=verbose
    )


def te(*args, **kwargs):
    """

    See Also
    --------
    TensorTerm : for developer details
    """
    return TensorTerm(*args, **kwargs)


intercept = Intercept()

# copy docs
for minimal_, class_ in zip(
    [l, s, f, te], [LinearTerm, SplineTerm, FactorTerm, TensorTerm]
):
    minimal_.__doc__ = class_.__init__.__doc__ + minimal_.__doc__


TERMS = {
    "term": Term,
    "intercept_term": Intercept,
    "linear_term": LinearTerm,
    "spline_term": SplineTerm,
    "factor_term": FactorTerm,
    "tensor_term": TensorTerm,
    "term_list": TermList,
}
