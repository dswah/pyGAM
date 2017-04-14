"""
Link functions
"""
from __future__ import division, absolute_import

import numpy as np

from pygam.core import Core


class Link(Core):
    def __init__(self, name=None):
        super(Link, self).__init__(name=name)

class IdentityLink(Link):
    def __init__(self):
        super(IdentityLink, self).__init__(name='identity')

    def link(self, mu, dist):
        """
        glm link function
        this is useful for going from mu to the linear prediction
        """
        return mu

    def mu(self, lp, dist):
        """
        glm mean function, ie inverse of link function
        this is useful for going from the linear prediction to mu
        """
        return lp

    def gradient(self, mu, dist):
        """
        derivative of the link function wrt mu
        """
        return np.ones_like(mu)

class LogitLink(Link):
    def __init__(self):
        super(LogitLink, self).__init__(name='logit')

    def link(self, mu, dist):
        """
        glm link function
        this is useful for going from mu to the linear prediction
        """
        return np.log(mu) - np.log(dist.levels - mu)

    def mu(self, lp, dist):
        """
        glm mean function, ie inverse of link function
        this is useful for going from the linear prediction to mu
        """
        elp = np.exp(lp)
        return dist.levels * elp / (elp + 1)

    def gradient(self, mu, dist):
        """
        derivative of the link function wrt mu
        """
        return dist.levels/(mu*(dist.levels - mu))

class LogLink(Link):
    def __init__(self):
        super(LogLink, self).__init__(name='log')

    def link(self, mu, dist):
        """
        glm link function
        this is useful for going from mu to the linear prediction
        """
        return np.log(mu)

    def mu(self, lp, dist):
        """
        glm mean function, ie inverse of link function
        this is useful for going from the linear prediction to mu
        """
        return np.exp(lp)

    def gradient(self, mu, dist):
        """
        derivative of the link function wrt mu
        """
        return 1. / mu

class InverseLink(Link):
    def __init__(self):
        super(InverseLink, self).__init__(name='inverse')

    def link(self, mu, dist):
        """
        glm link function
        this is useful for going from mu to the linear prediction
        """
        return mu**-1

    def mu(self, lp, dist):
        """
        glm mean function, ie inverse of link function
        this is useful for going from the linear prediction to mu
        """
        return lp**-1

    def gradient(self, mu, dist):
        """
        derivative of the link function wrt mu
        """
        return -1 * mu**-2

class InvSquaredLink(Link):
    def __init__(self):
        super(InvSquaredLink, self).__init__(name='inv_squared')

    def link(self, mu, dist):
        """
        glm link function
        this is useful for going from mu to the linear prediction
        """
        return mu**-2

    def mu(self, lp, dist):
        """
        glm mean function, ie inverse of link function
        this is useful for going from the linear prediction to mu
        """
        return lp**-0.5

    def gradient(self, mu, dist):
        """
        derivative of the link function wrt mu
        """
        return -2 * mu**-3
