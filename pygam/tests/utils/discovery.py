"""
The :mod:`pygam.tests.utils.discovery` module includes utilities to discover
objects (i.e. estimators, displays, functions) from the ``pygam`` package.
"""

# Adapted from scikit-learn
# Authors: scikit-learn-contrib developers
# License: BSD 3 clause

import inspect
import pkgutil
import sys
from importlib import import_module
from operator import itemgetter
from pathlib import Path

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.utils._testing import ignore_warnings

_MODULE_TO_IGNORE = {"tests"}
ROOT = Path(__file__).parents[2]

# pkgutil does not fully support Path objects until 3.11
if int(sys.version.split(".")[1]) < 11:
    ROOT = str(ROOT)


def all_estimators(type_filter=None):
    """Get a list of all estimators from ``pygam/``.

    This function crawls the module and gets all classes that inherit
    from ``BaseEstimator``. Classes that are defined in test-modules are not
    included.

    Parameters
    ----------
    type_filter : {"classifier", "regressor", "cluster", "transformer"} \
            or list of such str, default=None
        Which kind of estimators should be returned. If None, no filter is
        applied and all estimators are returned.  Possible values are
        'classifier', 'regressor', 'cluster' and 'transformer' to get
        estimators only of these specific types, or a list of these to
        get the estimators that fit at least one of the types.

    Returns
    -------
    estimators : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actual type of the class.

    Examples
    --------
    >>> from pygam.tests.utils.discovery import all_estimators
    >>> estimators = all_estimators()
    >>> type(estimators)
    <class 'list'>
    """

    def is_abstract(c):
        if not (hasattr(c, "__abstractmethods__")):
            return False
        if not len(c.__abstractmethods__):
            return False
        return True

    all_classes = []
    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with ignore_warnings(category=FutureWarning):
        for _, module_name, _ in pkgutil.walk_packages(path=[ROOT], prefix="pygam."):
            module_parts = module_name.split(".")
            if any(part in _MODULE_TO_IGNORE for part in module_parts):
                continue
            module = import_module(module_name)
            classes = inspect.getmembers(module, inspect.isclass)
            classes = [
                (name, est_cls) for name, est_cls in classes if not name.startswith("_")
            ]

            all_classes.extend(classes)

    all_classes = set(all_classes)

    estimators = [
        c
        for c in all_classes
        if (issubclass(c[1], BaseEstimator) and c[0] != "BaseEstimator")
    ]
    # get rid of abstract base classes
    estimators = [c for c in estimators if not is_abstract(c[1])]

    if type_filter is not None:
        if not isinstance(type_filter, list):
            type_filter = [type_filter]
        else:
            type_filter = list(type_filter)  # copy
        filtered_estimators = []
        filters = {
            "classifier": ClassifierMixin,
            "regressor": RegressorMixin,
            "transformer": TransformerMixin,
            "cluster": ClusterMixin,
        }
        for name, mixin in filters.items():
            if name in type_filter:
                type_filter.remove(name)
                filtered_estimators.extend(
                    [est for est in estimators if issubclass(est[1], mixin)]
                )
        estimators = filtered_estimators
        if type_filter:
            raise ValueError(
                "Parameter type_filter must be 'classifier', "
                "'regressor', 'transformer', 'cluster' or "
                "None, got"
                f" {repr(type_filter)}."
            )

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(estimators), key=itemgetter(0))


def all_displays():
    """Get a list of all displays from ``pygam``.

    Returns
    -------
    displays : list of tuples
        List of (name, class), where ``name`` is the display class name as
        string and ``class`` is the actual type of the class.

    Examples
    --------
    >>> from pygam.tests.utils.discovery import all_displays
    >>> displays = all_displays()
    """
    all_classes = []
    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with ignore_warnings(category=FutureWarning):
        for _, module_name, _ in pkgutil.walk_packages(path=[ROOT], prefix="pygam."):
            module_parts = module_name.split(".")
            if any(part in _MODULE_TO_IGNORE for part in module_parts):
                continue
            module = import_module(module_name)
            classes = inspect.getmembers(module, inspect.isclass)
            classes = [
                (name, display_class)
                for name, display_class in classes
                if not name.startswith("_") and name.endswith("Display")
            ]
            all_classes.extend(classes)

    return sorted(set(all_classes), key=itemgetter(0))


def _is_checked_function(item):
    if not inspect.isfunction(item):
        return False

    if item.__name__.startswith("_"):
        return False

    mod = item.__module__
    if not mod.startswith("pygam.") or mod.endswith("estimator_checks"):
        return False

    return True


def all_functions():
    """Get a list of all functions from ``pygam``.

    Returns
    -------
    functions : list of tuples
        List of (name, function), where ``name`` is the function name as
        string and ``function`` is the actual function.

    Examples
    --------
    >>> from pygam.tests.utils.discovery import all_functions
    >>> functions = all_functions()
    """
    all_functions = []
    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with ignore_warnings(category=FutureWarning):
        for _, module_name, _ in pkgutil.walk_packages(path=[ROOT], prefix="pygam."):
            module_parts = module_name.split(".")
            if any(part in _MODULE_TO_IGNORE for part in module_parts):
                continue

            module = import_module(module_name)
            functions = inspect.getmembers(module, _is_checked_function)
            functions = [
                (func.__name__, func)
                for name, func in functions
                if not name.startswith("_")
            ]
            all_functions.extend(functions)

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(all_functions), key=itemgetter(0))
