"""Core classes."""

import numpy as np

from pygam.utils import flatten, round_to_n_decimal_places


def nice_repr(
    name,
    param_kvs,
    line_width=30,
    line_offset=5,
    decimals=3,
    args=None,
    flatten_attrs=True,
):
    """
    Tool to do a nice repr of a class.

    Parameters
    ----------
    name : str
        class name
    param_kvs : dict
        dict containing class parameters names as keys,
        and the corresponding values as values
    line_width : int
        desired maximum line width.
        default: 30
    line_offset : int
        desired offset for new lines
        default: 5
    decimals : int
        number of decimal places to keep for float values
        default: 3

    Returns
    -------
    out : str
        nicely formatted repr of class instance
    """
    if not param_kvs and not args:
        # if the object has no params it's easy
        return f"{name}()"

    # sort keys and values
    ks = list(param_kvs.keys())
    vs = list(param_kvs.values())
    idxs = np.argsort(ks)
    param_kvs = [(ks[i], vs[i]) for i in idxs]

    if args is not None:
        param_kvs = [(None, arg) for arg in args] + param_kvs

    param_kvs = param_kvs[::-1]
    out = ""
    current_line = name + "("
    while len(param_kvs) > 0:
        # flatten sub-term properties, but not `terms`
        k, v = param_kvs.pop()
        if flatten_attrs and k != "terms":
            v = flatten(v)

        # round the floats first
        if issubclass(v.__class__, (float, np.ndarray)):
            v = round_to_n_decimal_places(v, n=decimals)
            v = str(v)
        else:
            v = repr(v)

        # handle args
        if k is None:
            param = f"{v},"
        else:
            param = f"{k}={v},"

        # print
        if len(current_line + param) <= line_width:
            current_line += param
        else:
            out += current_line + "\n"
            current_line = " " * line_offset + param

        if len(current_line) < line_width and len(param_kvs) > 0:
            current_line += " "

    out += current_line[:-1]  # remove trailing comma
    out += ")"
    return out


class Core:
    """
    Creates an instance of the Core class.

    comes loaded with useful methods

    Parameters
    ----------
    name : str, default: None
    line_width : int, default: 70
        number of characters to print on a line
    line_offset : int, default: 3
        number of characters to indent after the first line

    Returns
    -------
    self
    """

    def __init__(self, name=None, line_width=70, line_offset=3):
        self._name = name
        self._line_width = line_width
        self._line_offset = line_offset

        if not hasattr(self, "_exclude"):
            self._exclude = []

        if not hasattr(self, "_include"):
            self._include = []

    def __str__(self):
        """__str__ method."""
        if self._name is None:
            return self.__repr__()
        return self._name

    def __repr__(self):
        """__repr__ method."""
        name = self.__class__.__name__
        return nice_repr(
            name,
            self.get_params(),
            line_width=self._line_width,
            line_offset=self._line_offset,
            decimals=4,
            args=None,
        )

    def get_params(self, deep=False):
        """
        Returns a dict of all of the object's user-facing parameters.

        Parameters
        ----------
        deep : boolean, default: False
            when True, also gets non-user-facing parameters

        Returns
        -------
        dict
        """
        attrs = self.__dict__
        for attr in self._include:
            attrs[attr] = getattr(self, attr)

        if deep is True:
            return attrs
        return dict(
            [
                (k, v)
                for k, v in list(attrs.items())
                if (k[0] != "_") and (k[-1] != "_") and (k not in self._exclude)
            ]
        )

    def set_params(self, deep=False, force=False, **parameters):
        """
        Sets an object's parameters.

        Parameters
        ----------
        deep : boolean, default: False
            when True, also sets non-user-facing parameters
        force : boolean, default: False
            when True, also sets parameters that the object does not already
            have
        **parameters : parameters to set

        Returns
        -------
        self
        """
        param_names = self.get_params(deep=deep).keys()
        for parameter, value in parameters.items():
            if (
                parameter in param_names
                or force
                or (hasattr(self, parameter) and parameter == parameter.strip("_"))
            ):
                setattr(self, parameter, value)
        return self


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
