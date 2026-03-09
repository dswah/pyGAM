import numpy as np

from pygam.distributions import BinomialDist
from pygam.links import (
    CLogLogLink,
    IdentityLink,
    InverseLink,
    InvSquaredLink,
    LogitLink,
    LogLink,
)


def check_inverse(link):
    dist = BinomialDist(levels=1)

    mu = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    lp = link.link(mu, dist)
    mu_recovered = link.mu(lp, dist)

    np.testing.assert_allclose(mu, mu_recovered, rtol=1e-5)


def test_identity_link_inverse():
    check_inverse(IdentityLink())


def test_log_link_inverse():
    check_inverse(LogLink())


def test_logit_link_inverse():
    check_inverse(LogitLink())


def test_inverse_link_inverse():
    check_inverse(InverseLink())


def test_inv_squared_link_inverse():
    check_inverse(InvSquaredLink())


def test_cloglog_link_inverse():
    check_inverse(CLogLogLink())
