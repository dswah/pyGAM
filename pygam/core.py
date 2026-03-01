"""Core Classes"""

import inspect

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

    @classmethod
    def _get_param_names(cls):
        """Get parameter names from ``__init__`` signatures across the MRO.

        Walks the class hierarchy collecting every explicit parameter declared
        in ``__init__``, skipping ``self``, ``*args``, and ``**kwargs``.

        Returns
        -------
        list of str
            Sorted parameter names.
        """
        params = []
        for klass in cls.__mro__:
            if klass is object:
                continue
            init = getattr(klass, "__init__", None)
            if init is None or init is object.__init__:
                continue
            try:
                sig = inspect.signature(init)
            except (ValueError, TypeError):
                continue
            for name, param in sig.parameters.items():
                if name == "self":
                    continue
                if param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
                    continue
                if name not in params:
                    params.append(name)
        return sorted(params)

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
        if deep is True:
            # Legacy: return every instance attribute plus _include additions
            attrs = dict(self.__dict__)
            for attr in self._include:
                attrs[attr] = getattr(self, attr)
            return attrs

        # Discover params declared in __init__ signatures
        param_names = list(self._get_param_names())

        # Pick up attributes from _plural that were explicitly set on the
        # instance via **kwargs (e.g. lam passed to GAM()).
        for attr in getattr(self, "_plural", []):
            if attr not in param_names and attr in self.__dict__:
                if attr[0] != "_" and attr[-1] != "_":
                    param_names.append(attr)

        # Add any explicitly registered extra attributes
        for attr in getattr(self, "_include", []):
            if attr not in param_names:
                param_names.append(attr)

        # Remove excluded attributes
        exclude = getattr(self, "_exclude", [])
        param_names = [p for p in param_names if p not in exclude]

        # Only return attributes that actually exist on the instance
        return {
            name: getattr(self, name) for name in param_names if name in self.__dict__
        }

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
