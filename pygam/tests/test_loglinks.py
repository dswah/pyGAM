import numpy as np

from pygam.links import LogLink


def test_loglink_extreme_lp_no_overflow():
    link = LogLink()
    lp = np.array([-1000.0, 1000.0, 100000.0, -100000.0])

    # Only raise on overflow inside this block
    with np.errstate(over="raise"):
        mu = link.mu(lp, None)
        grad = link.gradient(mu, None)

    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(grad))
