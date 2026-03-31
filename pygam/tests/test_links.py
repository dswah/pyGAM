def test_log_link_no_overflow():
    """LogLink.mu should not overflow for large linear predictor values."""
    from pygam.links import LogLink
    import numpy as np
    link = LogLink()
    lp = np.array([-1000.0, 0.0, 1000.0])
    result = link.mu(lp, dist=None)
    assert np.all(np.isfinite(result)), "LogLink.mu produced inf or nan"

def test_logit_link_no_overflow():
    """LogitLink.mu should not overflow for large linear predictor values."""
    from pygam.links import LogitLink
    from pygam.distributions import BinomialDist
    import numpy as np
    link = LogitLink()
    dist = BinomialDist()
    lp = np.array([-1000.0, 0.0, 1000.0])
    result = link.mu(lp, dist)
    assert np.all(np.isfinite(result)), "LogitLink.mu produced inf or nan"
