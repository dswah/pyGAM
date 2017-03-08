"""
Core classes
"""

import numpy as np

from utils import round_to_n_decimal_places

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
        return '%s()' % name

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
    """
    core class

    comes loaded with useful methods
    """
    def __init__(self, name=None, line_width=70, line_offset=3):
        self._name = name
        self._line_width = line_width
        self._line_offset = line_offset
        self._exclude = []

    def __str__(self):
        if self._name is None:
            return self.__repr__()
        return self._name

    def __repr__(self):
        name = self.__class__.__name__
        param_kvs = [(k,v) for k,v in self.get_params().iteritems()]

        return nice_repr(name, param_kvs, line_width=self._line_width, line_offset=self._line_offset)

    def get_params(self):
        return dict([(k,v) for k,v in self.__dict__.iteritems() if k[0]!='_' and (k not in self._exclude)])

    def set_params(self, **parameters):
        param_names = self.get_params().keys()
        for parameter, value in parameters.items():
            if parameter in param_names:
                setattr(self, parameter, value)
        return self
