"""
Link functions
"""

import numpy as np

from core import Core


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
        glm mean ie inverse of link function
        """
        return lp

    def gradient(self, mu, dist):
        """
        derivative of the linear prediction wrt mu
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
        return np.log(mu / (dist.levels - mu))

    def mu(self, lp, dist):
        """
        glm mean ie inverse of link function
        for classification this is the prediction probabilities
        """
        elp = np.exp(lp)
        return dist.levels * elp / (elp + 1)

    def gradient(self, mu, dist):
        """
        derivative of the linear prediction wrt mu
        """
        return dist.levels/(mu*(dist.levels - mu))
