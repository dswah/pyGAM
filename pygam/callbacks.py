"""CallBacks"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any

import numpy as np

from pygam.core import Core

if TYPE_CHECKING:
    from pygam.pygam import GAM


def validate_callback_data(method: Callable) -> Callable:
    """
    Wraps a callback's method to pull the desired arguments from the vars dict
    also checks to ensure the method's arguments are in the vars dict.

    Parameters
    ----------
    method : callable

    Returns
    -------
    validated callable
    """

    @wraps(method)
    def method_wrapper(*args: Any, **kwargs: Any) -> Any:
        """

        Parameters
        ----------
        *args
        **kwargs

        Returns
        -------
        method's output
        """
        expected = method.__code__.co_varnames

        # rename current gam object
        if "self" in kwargs:
            gam = kwargs["self"]
            del kwargs["self"]
            kwargs["gam"] = gam

        # loop once to check any missing
        missing = []
        for e in expected:
            if e == "self":
                continue
            if e not in kwargs:
                missing.append(e)
        assert len(missing) == 0, "CallBack cannot reference: {}".format(
            ", ".join(missing)
        )

        # loop again to extract desired
        kwargs_subset = {}
        for e in expected:
            if e == "self":
                continue
            kwargs_subset[e] = kwargs[e]

        return method(*args, **kwargs_subset)

    return method_wrapper


def validate_callback(callback: type[CallBack]) -> type[CallBack]:
    """
    Validates a callback's on_loop_start and on_loop_end methods.

    Parameters
    ----------
    callback : Callback object

    Returns
    -------
    validated callback
    """
    if (
        not (hasattr(callback, "_validated"))
        or getattr(callback, "_validated") is False
    ):
        assert hasattr(callback, "on_loop_start") or hasattr(callback, "on_loop_end"), (
            "callback must have `on_loop_start` or `on_loop_end` method"
        )
        if hasattr(callback, "on_loop_start"):
            setattr(
                callback,
                "on_loop_start",
                validate_callback_data(getattr(callback, "on_loop_start")),
            )
        if hasattr(callback, "on_loop_end"):
            setattr(
                callback,
                "on_loop_end",
                validate_callback_data(getattr(callback, "on_loop_end")),
            )
        setattr(callback, "_validated", True)
    return callback


class CallBack(Core):
    """
    Base CallBack

    Parameters
    ----------
    """

    def __init__(self, name: str | None = None) -> None:
        super(CallBack, self).__init__(name=name)


@validate_callback
class Deviance(CallBack):
    """
    Deviance CallBack class

    Useful for capturing the Deviance of a model on training data
    at each iteration

    Parameters
    ----------
    """

    def __init__(self) -> None:
        super(Deviance, self).__init__(name="deviance")

    def on_loop_start(self, gam: GAM, y: np.ndarray, mu: np.ndarray) -> float:
        """
        Runs the method at loop start.

        Parameters
        ----------
        gam : GAM instance
        y : array-like of length n
            target data
        mu : array-like of length n
            expected value data

        Returns
        -------
        deviance : np.array of length n
        """
        return float(gam.distribution.deviance(y=y, mu=mu, scaled=False).sum())


@validate_callback
class Accuracy(CallBack):
    """
    Accuracy CallBack

    Useful for capturing the accuracy of a model on training data
    at each iteration

    Parameters
    ----------
    """

    def __init__(self) -> None:
        super(Accuracy, self).__init__(name="accuracy")

    def on_loop_start(self, y: np.ndarray, mu: np.ndarray) -> float:
        """
        Runs the method at start of each optimization loop.

        Parameters
        ----------
        y : array-like of length n
            target data
        mu : array-like of length n
            expected value data

        Returns
        -------
        accuracy : np.array of length n
        """
        return float(np.mean(y == (mu > 0.5)))


@validate_callback
class Diffs(CallBack):
    """
    Differences Callback

    Useful for capturing the differences in model coefficient norms between
    iterations

    Parameters
    ----------
    """

    def __init__(self) -> None:
        super(Diffs, self).__init__(name="diffs")

    def on_loop_end(self, diff: float) -> float:
        """
        Runs the method at end of each optimization loop.

        Parameters
        ----------
        diff : float

        Returns
        -------
        diff : float
        """
        return diff


@validate_callback
class Coef(CallBack):
    """
    Coefficients CallBack

    Useful for capturing the models coefficients at each iteration

    Parameters
    ----------
    """

    def __init__(self) -> None:
        super(Coef, self).__init__(name="coef")

    def on_loop_start(self, gam: GAM) -> np.ndarray:
        """
        Runs the method at start of each optimization loop.

        Parameters
        ----------
        gam : GAM instance

        Returns
        -------
        coef_ : np.ndarray
        """
        return gam.coef_


CALLBACKS: dict[str, type[CallBack]] = {
    "deviance": Deviance,
    "diffs": Diffs,
    "accuracy": Accuracy,
    "coef": Coef,
}
