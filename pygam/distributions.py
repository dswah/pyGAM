"""
Distributions
"""
from __future__ import division

import scipy as sp
import numpy as np

from core import Core
from utils import ylogydu


class Distribution(Core):
    """
    base distribution class
    """
    def __init__(self, name=None, scale=None):
        self.scale = scale
        self._known_scale = self.scale is not None
        super(Distribution, self).__init__(name=name)
        if not self._known_scale:
            self._exclude += ['scale']

    def phi(self, y, mu, edof):
        """
        GLM scale parameter.
        for Binomial and Poisson families this is unity
        for Normal family this is variance
        """
        if self._known_scale:
            return self.scale
        else:
            return np.sum(self.V(mu**-1) * (y - mu)**2) / (len(mu) - edof)

class NormalDist(Distribution):
    """
    Normal Distribution
    """
    def __init__(self, scale=None):
        super(NormalDist, self).__init__(name='normal', scale=scale)

    def pdf(self, y, mu):
        return np.exp(-(y - mu)**2/(2*self.scale)) / (self.scale * 2 * np.pi)**0.5

    def V(self, mu):
        """glm Variance function"""
        return np.ones_like(mu)

    def deviance(self, y, mu, scaled=True, summed=True):
        """
        model deviance

        for a gaussian linear model, this is equal to the SSE
        """
        dev = (y - mu)**2
        if scaled:
            dev /= self.scale
        if summed:
            return dev.sum()
        return dev

class BinomialDist(Distribution):
    """
    Binomial Distribution
    """
    def __init__(self, levels=1):
        if levels is None:
            levels = 1
        self.levels = levels
        super(BinomialDist, self).__init__(name='binomial', scale=1.)
        self._exclude.append('scale')

    def pdf(self, y, mu):
        n = self.levels
        return (sp.misc.comb(n, y) * (mu / n)**y * (1 - (mu / n))**(n - y))

    def V(self, mu):
        """glm Variance function"""
        return mu * (1 - mu/self.levels)

    def deviance(self, y, mu, scaled=True, summed=True):
        """
        model deviance

        for a bernoulli logistic model, this is equal to the twice the negative loglikelihod.
        """
        dev = 2 * (ylogydu(y, mu) + ylogydu(self.levels - y, self.levels-mu))
        if scaled:
            dev /= self.scale
        if summed:
            return dev.sum()
        return dev

class PoissonDist(Distribution):
    """
    Poisson Distribution
    """
    def __init__(self):
        super(PoissonDist, self).__init__(name='poisson', scale=1.)
        self._exclude.append('scale')

    def pdf(self, y, mu):
        return (mu**y) * np.exp(-mu) / sp.misc.factorial(y)

    def V(self, mu):
        """glm Variance function"""
        return mu

    def deviance(self, y, mu, scaled=True, summed=True):
        """
        model deviance

        for a bernoulli logistic model, this is equal to the twice the negative loglikelihod.
        """
        dev = 2 * (ylogydu(y, mu) - (y - mu))

        if scaled:
            dev /= self.scale
        if summed:
            return dev.sum()
        return dev

class GammaDist(Distribution):
    """
    Gamma Distribution
    """
    def __init__(self, scale=None):
        super(GammaDist, self).__init__(name='gamma', scale=scale)

    def pdf(self, y, mu):
        nu = 1./self.scale
        return 1./sp.special.gamma(nu) * (nu/mu)**nu * y**(nu-1) * np.exp(-nu * y / mu)

    def V(self, mu):
        """glm Variance function"""
        return mu**2

    def deviance(self, y, mu, scaled=True, summed=True):
        """
        model deviance

        for a bernoulli logistic model, this is equal to the twice the negative loglikelihod.
        """
        dev = 2 * ((y - mu)/mu - np.log(y/mu))

        if scaled:
            dev /= self.scale
        if summed:
            return dev.sum()
        return dev

class InvGaussDist(Distribution):
    """
    Inverse Gaussian Distribution
    """
    def __init__(self, scale=None):
        super(InvGaussDist, self).__init__(name='inv_gauss', scale=scale)

    def pdf(self, y, mu):
        gamma = 1./self.scale
        return (gamma / (2 * np.pi * y**3))**.5 * np.exp(-gamma * (y - mu)**2 / (2 * mu**2 * y))

    def V(self, mu):
        """glm Variance function"""
        return mu**3

    def deviance(self, y, mu, scaled=True, summed=True):
        """
        model deviance

        for a bernoulli logistic model, this is equal to the twice the negative loglikelihod.
        """
        dev = ((y - mu)**2) / (mu**2 * y)

        if scaled:
            dev /= self.scale
        if summed:
            return dev.sum()
        return dev
