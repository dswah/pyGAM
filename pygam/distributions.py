"""
Distributions
"""

from __future__ import division, absolute_import

import scipy as sp
import numpy as np

from pygam.core import Core
from pygam.utils import ylogydu


class Distribution(Core):
    """
    base distribution class
    """
    def __init__(self, name=None, scale=None):
        """
        creates an instance of the Distribution class

        Parameters
        ----------
        name : str, default: None
        scale : float or None, default: None
            scale/standard deviation of the distribution

        Returns
        -------
        self
        """
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

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        edof : float
            estimated degrees of freedom

        Returns
        -------
        scale : estimated model scale
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
        """
        creates an instance of the NormalDist class

        Parameters
        ----------
        scale : float or None, default: None
            scale/standard deviation of the distribution

        Returns
        -------
        self
        """
        super(NormalDist, self).__init__(name='normal', scale=scale)

    def pdf(self, y, mu):
        """
        computes the pdf or pmf of the values under the current distribution

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values

        Returns
        -------
        pdf/pmf : np.array of length n
        """
        return np.exp(-(y - mu)**2/(2*self.scale)) / (self.scale * 2 * np.pi)**0.5

    def V(self, mu):
        """
        glm Variance function

        computes the variance of the distribtion

        Parameters
        ----------
        mu : array-like of length n
            expected values

        Returns
        -------
        variance : np.array of length n
        """
        return np.ones_like(mu)

    def deviance(self, y, mu, scaled=True, summed=True):
        """
        model deviance

        for a gaussian linear model, this is equal to the SSE

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        scaled : boolean, default: True
            whether to divide the deviance by the distribution scaled
        summed : boolean, default: True
            whether to sum the deviances

        Returns
        -------
        deviances : np.array of length n
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
        """
        creates an instance of the NormalDist class

        Parameters
        ----------
        levels : int of None, default: 1
            number of levels in the binomial distribution

        Returns
        -------
        self
        """
        if levels is None:
            levels = 1
        self.levels = levels
        super(BinomialDist, self).__init__(name='binomial', scale=1.)
        self._exclude.append('scale')

    def pdf(self, y, mu):
        """
        computes the pdf or pmf of the values under the current distribution

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values

        Returns
        -------
        pdf/pmf : np.array of length n
        """
        n = self.levels
        return (sp.misc.comb(n, y) * (mu / n)**y * (1 - (mu / n))**(n - y))

    def V(self, mu):
        """
        glm Variance function

        computes the variance of the distribtion

        Parameters
        ----------
        mu : array-like of length n
            expected values

        Returns
        -------
        variance : np.array of length n
        """
        return mu * (1 - mu/self.levels)

    def deviance(self, y, mu, scaled=True, summed=True):
        """
        model deviance

        for a bernoulli logistic model, this is equal to the twice the
        negative loglikelihod.

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        scaled : boolean, default: True
            whether to divide the deviance by the distribution scaled
        summed : boolean, default: True
            whether to sum the deviances

        Returns
        -------
        deviances : np.array of length n
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
        """
        creates an instance of the PoissonDist class

        Parameters
        ----------
        None

        Returns
        -------
        self
        """
        super(PoissonDist, self).__init__(name='poisson', scale=1.)
        self._exclude.append('scale')

    def pdf(self, y, mu):
        """
        computes the pdf or pmf of the values under the current distribution

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values

        Returns
        -------
        pdf/pmf : np.array of length n
        """
        return (mu**y) * np.exp(-mu) / sp.misc.factorial(y)

    def V(self, mu):
        """
        glm Variance function

        computes the variance of the distribtion

        Parameters
        ----------
        mu : array-like of length n
            expected values

        Returns
        -------
        variance : np.array of length n
        """
        return mu

    def deviance(self, y, mu, scaled=True, summed=True):
        """
        model deviance

        for a bernoulli logistic model, this is equal to the twice the
        negative loglikelihod.

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        scaled : boolean, default: True
            whether to divide the deviance by the distribution scaled
        summed : boolean, default: True
            whether to sum the deviances

        Returns
        -------
        deviances : np.array of length n
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
        """
        creates an instance of the GammaDist class

        Parameters
        ----------
        scale : float or None, default: None
            scale/standard deviation of the distribution

        Returns
        -------
        self
        """
        super(GammaDist, self).__init__(name='gamma', scale=scale)

    def pdf(self, y, mu):
        """
        computes the pdf or pmf of the values under the current distribution

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values

        Returns
        -------
        pdf/pmf : np.array of length n
        """
        nu = 1./self.scale
        return 1./sp.special.gamma(nu) * (nu/mu)**nu * y**(nu-1) * np.exp(-nu * y / mu)

    def V(self, mu):
        """
        glm Variance function

        computes the variance of the distribtion

        Parameters
        ----------
        mu : array-like of length n
            expected values

        Returns
        -------
        variance : np.array of length n
        """
        return mu**2

    def deviance(self, y, mu, scaled=True, summed=True):
        """
        model deviance

        for a bernoulli logistic model, this is equal to the twice the
        negative loglikelihod.

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        scaled : boolean, default: True
            whether to divide the deviance by the distribution scaled
        summed : boolean, default: True
            whether to sum the deviances

        Returns
        -------
        deviances : np.array of length n
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
        """
        creates an instance of the NormalDist class

        Parameters
        ----------
        scale : float or None, default: None
            scale/standard deviation of the distribution

        Returns
        -------
        self
        """
        super(InvGaussDist, self).__init__(name='inv_gauss', scale=scale)

    def pdf(self, y, mu):
        """
        computes the pdf or pmf of the values under the current distribution

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values

        Returns
        -------
        pdf/pmf : np.array of length n
        """
        gamma = 1./self.scale
        return (gamma / (2 * np.pi * y**3))**.5 * np.exp(-gamma * (y - mu)**2 / (2 * mu**2 * y))

    def V(self, mu):
        """
        glm Variance function

        computes the variance of the distribtion

        Parameters
        ----------
        mu : array-like of length n
            expected values

        Returns
        -------
        variance : np.array of length n
        """
        return mu**3

    def deviance(self, y, mu, scaled=True, summed=True):
        """
        model deviance

        for a bernoulli logistic model, this is equal to the twice the
        negative loglikelihod.

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        scaled : boolean, default: True
            whether to divide the deviance by the distribution scaled
        summed : boolean, default: True
            whether to sum the deviances

        Returns
        -------
        deviances : np.array of length n
        """
        dev = ((y - mu)**2) / (mu**2 * y)

        if scaled:
            dev /= self.scale
        if summed:
            return dev.sum()
        return dev
