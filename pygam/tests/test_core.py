# -*- coding: utf-8 -*-

from pygam.core import Core, nice_repr


def test_Core_class():
    """
    test attributes of core class
    """
    c = Core()
    assert c._name is None

    c = Core(name='cat', line_width=70, line_offset=3)
    assert c._name == 'cat'
    assert c._line_width == 70
    assert c._line_offset == 3


def test_nice_repr():
    """
    test a simple repr for a fake object
    """
    param_kvs = {}
    out = nice_repr('hi', param_kvs, line_width=30, line_offset=5, decimals=3)
    assert out == "hi()"


def test_nice_repr_more_attrs():
    """
    test a simple repr for a fake object with more attrs
    """
    param_kvs = {'color': 'blue', 'n_ears': 3, 'height': 1.3336}
    out = nice_repr('hi', param_kvs, line_width=60, line_offset=5, decimals=3)
    assert out == "hi(color='blue', height=1.334, n_ears=3)"
