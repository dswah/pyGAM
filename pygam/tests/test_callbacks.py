import numpy as np
import pytest

from pygam.callbacks import CallBack, validate_callback


class BadCallback(CallBack):
    def __init__(self):
        super(BadCallback, self).__init__(name="bad")

    def on_loop_start(self, missing_var):
        return missing_var


class GoodCallback(CallBack):
    def __init__(self):
        super(GoodCallback, self).__init__(name="good")

    def on_loop_start(self, y, mu):
        return (y + mu).sum()


def test_validate_callback_data_raises_value_error_on_missing_variable():
    callback = validate_callback(BadCallback())

    with pytest.raises(ValueError, match="CallBack cannot reference: missing_var"):
        callback.on_loop_start(y=np.array([1.0]), mu=np.array([2.0]))


def test_validate_callback_data_accepts_expected_variables_and_ignores_extra():
    callback = validate_callback(GoodCallback())

    result = callback.on_loop_start(
        y=np.array([1.0]),
        mu=np.array([2.0]),
        irrelevant=np.array([999.0]),
    )

    assert result == 3.0
