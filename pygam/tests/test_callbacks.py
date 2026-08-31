import pytest

from pygam import LinearGAM
from pygam.callbacks import CallBack, validate_callback_data


class _LocalVarCallback(CallBack):
    """A minimal custom callback that declares a local variable inside
    on_loop_start. Regression test for validate_callback_data
    incorrectly treating local variables as required arguments.
    """

    def on_loop_start(self, gam):
        # this local variable should NOT be treated as a required
        # keyword argument by validate_callback_data
        step_threshold = 0.05
        return step_threshold


def test_validate_callback_data_ignores_local_variables():
    """
    check that a callback method's internal local variables are not
    mistaken for required arguments pulled from the training loop's
    variables dict
    """
    wrapped = validate_callback_data(_LocalVarCallback.on_loop_start)
    # should not raise AssertionError: CallBack cannot reference: step_threshold
    result = wrapped(_LocalVarCallback(), gam=None)
    assert result == 0.05


def test_validate_callback_data_still_flags_missing_arguments():
    """
    check that validate_callback_data still enforces that genuine
    formal parameters are supplied
    """

    class NeedsY(CallBack):
        def on_loop_start(self, gam, y):
            return y

    wrapped = validate_callback_data(NeedsY.on_loop_start)
    with pytest.raises(AssertionError, match="CallBack cannot reference: y"):
        wrapped(NeedsY(), gam=None)


def test_gam_fit_with_custom_local_var_callback(mcycle_X_y):
    """
    end-to-end check that a GAM can be fit with a custom callback that
    declares local variables, without raising
    """
    X, y = mcycle_X_y
    gam = LinearGAM(callbacks=[_LocalVarCallback()])
    gam.fit(X, y)
    assert gam._is_fitted
    assert len(gam.logs_[str(gam.callbacks[-1])]) > 0
