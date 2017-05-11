"""
Core classes
"""

from __future__ import absolute_import

import numpy as np

from pygam.utils import round_to_n_decimal_places

def nice_repr(name, param_kvs, line_width=30, line_offset=5, decimals=3):
    """
    tool to do a nice repr of a class.

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
    if len(param_kvs) == 0:
        # if the object has no params it's easy
        return '{}()'.format(name)

    # sort keys and values
    ks = list(param_kvs.keys())
    vs = list(param_kvs.values())
    idxs = np.argsort(ks)
    param_kvs = [(ks[i],vs[i]) for i in idxs]

    param_kvs = param_kvs[::-1]
    out = ''
    current_line = name + '('
    while len(param_kvs) > 0:
        k, v = param_kvs.pop()
        if issubclass(v.__class__, (float, np.ndarray)):
            # round the floats first
            v = round_to_n_decimal_places(v, n=decimals)
            param = '{}={},'.format(k, str(v))
        else:
            param = '{}={},'.format(k, repr(v))
        if len(current_line + param) <= line_width:
            current_line += param
        else:
            out += current_line + '\n'
            current_line = ' '*line_offset + param

        if len(current_line) < line_width and len(param_kvs) > 0:
            current_line += ' '

    out += current_line[:-1] # remove trailing comma
    out += ')'
    return out


class Core(object):
    def __init__(self, name=None, line_width=70, line_offset=3):
        """
        creates an instance of the Core class

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
        self._name = name
        self._line_width = line_width
        self._line_offset = line_offset
        self._exclude = []

    def __str__(self):
        """__str__ method"""
        if self._name is None:
            return self.__repr__()
        return self._name

    def __repr__(self):
        """__repr__ method"""
        name = self.__class__.__name__
        return nice_repr(name, self.get_params(),
                         line_width=self._line_width,
                         line_offset=self._line_offset,
                         decimals=4)

    def get_params(self, deep=False):
        """
        returns a dict of all of the object's user-facing parameters

        Parameters
        ----------
        deep : boolean, default: False
            when True, also gets non-user-facing paramters

        Returns
        -------
        dict
        """
        if deep is True:
            return self.__dict__
        return dict([(k,v) for k,v in list(self.__dict__.items()) \
                     if (k[0] != '_') \
                    and (k[-1] != '_') and (k not in self._exclude)])

    def set_params(self, deep=False, force=False, **parameters):
        """
        sets an object's paramters

        Parameters
        ----------
        deep : boolean, default: False
            when True, also sets non-user-facing paramters
        force : boolean, default: False
            when True, also sets parameters that the object does not already
            have
        **parameters : paramters to set

        Returns
        ------
        self
        """
        param_names = self.get_params(deep=deep).keys()
        for parameter, value in parameters.items():
            if (parameter in param_names) or force:
                setattr(self, parameter, value)
        return self
