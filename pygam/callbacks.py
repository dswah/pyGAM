"""
CallBacks
"""

import numpy as np

from core import Core


def validate_callback_data(method):
    def method_wrapper(*args, **kwargs):
        expected = method.__code__.co_varnames

        # rename curret gam object
        if 'self' in kwargs:
            gam = kwargs['self']
            del(kwargs['self'])
            kwargs['gam'] = gam

        # loop once to check any missing
        missing = []
        for e in expected:
            if e == 'self':
                continue
            if e not in kwargs:
                missing.append(e)
        assert len(missing) == 0, 'CallBack cannot reference: {}'.format(', '.join(missing))

        # loop again to extract desired
        kwargs_subset = {}
        for e in expected:
            if e == 'self':
                continue
            kwargs_subset[e] = kwargs[e]

        return method(*args, **kwargs_subset)

    return method_wrapper

def validate_callback(callback):
    if not(hasattr(callback, '_validated')) or callback._validated == False:
        assert hasattr(callback, 'on_loop_start') or hasattr(callback, 'on_loop_end'), 'callback must have `on_loop_start` or `on_loop_end` method'
        if hasattr(callback, 'on_loop_start'):
            setattr(callback, 'on_loop_start', validate_callback_data(callback.on_loop_start))
        if hasattr(callback, 'on_loop_end'):
            setattr(callback, 'on_loop_end', validate_callback_data(callback.on_loop_end))
        setattr(callback, '_validated', True)
    return callback


class CallBack(Core):
    def __init__(self, name):
        super(CallBack, self).__init__(name=name)


@validate_callback
class Deviance(CallBack):
    def __init__(self):
        super(Deviance, self).__init__(name='deviance')
    def on_loop_start(self, gam, y, mu):
        return gam.distribution.deviance(y=y, mu=mu, scaled=False)


@validate_callback
class Accuracy(CallBack):
    def __init__(self):
        super(Accuracy, self).__init__(name='accuracy')
    def on_loop_start(self, y, mu):
        return np.mean(y == (mu>0.5))


@validate_callback
class Diffs(CallBack):
    def __init__(self):
        super(Diffs, self).__init__(name='diffs')
    def on_loop_end(self, diff):
        return diff
