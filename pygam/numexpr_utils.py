"""Optional numexpr acceleration for element-wise math.

If numexpr is installed, provides fused simd evaluation of expressions.
Otherwise, falls back transparently to plain NumPy via ``eval()``.

Usage::

    from pygam.numexpr_utils import ne_evaluate
    result = ne_evaluate("a * log(b / c)", a=a, b=b, c=c)
"""

import numpy as np

try:
    import numexpr as _ne

    _HAS_NUMEXPR = True
except ImportError:
    _HAS_NUMEXPR = False


def ne_evaluate(expr, **kwargs):
    """Evaluate *expr* with numexpr if available, else  back to NumPy.

    Parameters
    ----------
    expr : str
        A numexpr-compatible expression string.
    **kwargs
        Named arrays / scalars referenced in *expr*.

    Returns
    -------
    np.ndarray
    """
    if _HAS_NUMEXPR:
        return _ne.evaluate(expr, local_dict=kwargs)

    # Fallback: inject numpy functions into the namespace so that
    # expressions like "log(x)" and "exp(x)" work with plain eval().
    ns = {
        "log": np.log,
        "exp": np.exp,
        "sqrt": np.sqrt,
        "abs": np.abs,
    }
    ns.update(kwargs)
    return eval(expr, {"__builtins__": {}}, ns)  # noqa: S307
