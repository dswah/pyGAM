"""Core Classes"""

import copy
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
    def __init__(self, name=None, line_width=70, line_offset=3):
        self._name = name
        self._line_width = line_width
        self._line_offset = line_offset

        if not hasattr(self, "_exclude"):
            self._exclude = []

        if not hasattr(self, "_include"):
            self._include = []

    def __str__(self):
        if self._name is None:
            return self.__repr__()
        return self._name

    def __repr__(self):
        name = self.__class__.__name__
        return nice_repr(
            name,
            self.get_params(),
            line_width=self._line_width,
            line_offset=self._line_offset,
            decimals=4,
            args=None,
        )

    # ✅ FIXED: NOW INSIDE CLASS
    def get_params(self, deep=False):
        attrs = self.__dict__.copy()

        for attr in self._include:
            attrs[attr] = getattr(self, attr)

        if deep:
            return copy.deepcopy(attrs)

        params = {
            k: v
            for k, v in attrs.items()
            if (k[0] != "_") and (k[-1] != "_") and (k not in self._exclude)
        }

        return copy.deepcopy(params)

    def set_params(self, deep=False, force=False, **parameters):
        param_names = self.get_params(deep=deep).keys()
        for parameter, value in parameters.items():
            if (
                parameter in param_names
                or force
                or (hasattr(self, parameter) and parameter == parameter.strip("_"))
            ):
                setattr(self, parameter, value)
        return self