"""
Distributions
"""

from functools import wraps
from abc import ABCMeta
from abc import abstractmethod

import scipy as sp
import numpy as np

from pygam.core import Core
from pygam.utils import ylogydu


def multiply_weights(deviance):
    @wraps(deviance)
    def multiplied(self, y, mu, weights=None, **kwargs):
        if weights is None:
            weights = np.ones_like(mu)
        return deviance(self, y, mu, **kwargs) * weights

    return multiplied


def divide_weights(V):
    @wraps(V)
    def divided(self, mu, weights=None, **kwargs):
        if weights is None:
            weights = np.ones_like(mu)
        return V(self, mu, **kwargs) / weights

    return divided


class Distribution(Core):
    __metaclass__ = ABCMeta
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

    def phi(self, y, mu, edof, weights):
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
        weights : array-like shape (n,) or None, default: None
            sample weights
            if None, defaults to array of ones

        Returns
        -------
        scale : estimated model scale
        """
        if self._known_scale:
            return self.scale
        else:
            return np.sum(weights * self.V(mu) ** -1 * (y - mu) ** 2) / (len(mu) - edof)

    @abstractmethod
    def sample(self, mu):
        """
        Return random samples from this distribution.

        Parameters
        ----------
        mu : array-like of shape n_samples or shape (n_simulations, n_samples)
            expected values

        Returns
        -------
        random_samples : np.array of same shape as mu
        """
        pass


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

    def log_pdf(self, y, mu, weights=None):
        """
        computes the log of the pdf or pmf of the values under the current distribution

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        weights : array-like shape (n,) or None, default: None
            sample weights
            if None, defaults to array of ones

        Returns
        -------
        pdf/pmf : np.array of length n
        """
        if weights is None:
            weights = np.ones_like(mu)
        scale = self.scale / weights
        return sp.stats.norm.logpdf(y, loc=mu, scale=scale)

    @divide_weights
    def V(self, mu):
        """
        glm Variance function.

        if
            Y ~ ExpFam(theta, scale=phi)
        such that
            E[Y] = mu = b'(theta)
        and
            Var[Y] = b''(theta) * phi / w

        then we seek V(mu) such that we can represent Var[y] as a fn of mu:
            Var[Y] = V(mu) * phi

        ie
            V(mu) = b''(theta) / w

        Parameters
        ----------
        mu : array-like of length n
            expected values

        Returns
        -------
        V(mu) : np.array of length n
        """
        return np.ones_like(mu)

    @multiply_weights
    def deviance(self, y, mu, scaled=True):
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

        Returns
        -------
        deviances : np.array of length n
        """
        dev = (y - mu) ** 2
        if scaled:
            dev /= self.scale
        return dev

    def sample(self, mu):
        """
        Return random samples from this Normal distribution.

        Samples are drawn independently from univariate normal distributions
        with means given by the values in `mu` and with standard deviations
        equal to the `scale` attribute if it exists otherwise 1.0.

        Parameters
        ----------
        mu : array-like of shape n_samples or shape (n_simulations, n_samples)
            expected values

        Returns
        -------
        random_samples : np.array of same shape as mu
        """
        standard_deviation = self.scale**0.5 if self.scale else 1.0
        return np.random.normal(loc=mu, scale=standard_deviation, size=None)


class BinomialDist(Distribution):
    """
    Binomial Distribution
    """

    def __init__(self, levels=1):
        """
        creates an instance of the Binomial class

        Parameters
        ----------
        levels : int of None, default: 1
            number of trials in the binomial distribution

        Returns
        -------
        self
        """
        if levels is None:
            levels = 1
        self.levels = levels
        super(BinomialDist, self).__init__(name='binomial', scale=1.0)
        self._exclude.append('scale')

    def log_pdf(self, y, mu, weights=None):
        """
        computes the log of the pdf or pmf of the values under the current distribution

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        weights : array-like shape (n,) or None, default: None
            sample weights
            if None, defaults to array of ones

        Returns
        -------
        pdf/pmf : np.array of length n
        """
        if weights is None:
            weights = np.ones_like(mu)
        n = self.levels
        p = mu / self.levels
        return sp.stats.binom.logpmf(y, n, p)

    @divide_weights
    def V(self, mu):
        """
        glm Variance function

        computes the variance of the distribution

        Parameters
        ----------
        mu : array-like of length n
            expected values

        Returns
        -------
        variance : np.array of length n
        """
        return mu * (1 - mu / self.levels)

    @multiply_weights
    def deviance(self, y, mu, scaled=True):
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

        Returns
        -------
        deviances : np.array of length n
        """
        dev = 2 * (ylogydu(y, mu) + ylogydu(self.levels - y, self.levels - mu))
        if scaled:
            dev /= self.scale
        return dev

    def sample(self, mu):
        """
        Return random samples from this Normal distribution.

        Parameters
        ----------
        mu : array-like of shape n_samples or shape (n_simulations, n_samples)
            expected values

        Returns
        -------
        random_samples : np.array of same shape as mu
        """
        number_of_trials = self.levels
        success_probability = mu / number_of_trials
        return np.random.binomial(n=number_of_trials, p=success_probability, size=None)


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
        super(PoissonDist, self).__init__(name='poisson', scale=1.0)
        self._exclude.append('scale')

    def log_pdf(self, y, mu, weights=None):
        """
        computes the log of the pdf or pmf of the values under the current distribution

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        weights : array-like shape (n,) or None, default: None
            containing sample weights
            if None, defaults to array of ones

        Returns
        -------
        pdf/pmf : np.array of length n
        """
        if weights is None:
            weights = np.ones_like(mu)
        # in Poisson regression weights are proportional to the exposure
        # so we want to pump up all our predictions
        # NOTE: we assume the targets are counts, not rate.
        # ie if observations were scaled to account for exposure, they have
        # been rescaled before calling this function.
        # since some samples have higher exposure,
        # they also need to have higher variance,
        # we do this by multiplying mu by the weight=exposure
        mu = mu * weights
        return sp.stats.poisson.logpmf(y, mu=mu)

    @divide_weights
    def V(self, mu):
        """
        glm Variance function

        computes the variance of the distribution

        Parameters
        ----------
        mu : array-like of length n
            expected values

        Returns
        -------
        variance : np.array of length n
        """
        return mu

    @multiply_weights
    def deviance(self, y, mu, scaled=True):
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

        Returns
        -------
        deviances : np.array of length n
        """
        dev = 2 * (ylogydu(y, mu) - (y - mu))

        if scaled:
            dev /= self.scale
        return dev

    def sample(self, mu):
        """
        Return random samples from this Poisson distribution.

        Parameters
        ----------
        mu : array-like of shape n_samples or shape (n_simulations, n_samples)
            expected values

        Returns
        -------
        random_samples : np.array of same shape as mu
        """
        return np.random.poisson(lam=mu, size=None)


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

    def log_pdf(self, y, mu, weights=None):
        """
        computes the log of the pdf or pmf of the values under the current distribution

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        weights : array-like shape (n,) or None, default: None
            containing sample weights
            if None, defaults to array of ones

        Returns
        -------
        pdf/pmf : np.array of length n
        """
        if weights is None:
            weights = np.ones_like(mu)
        nu = weights / self.scale
        return sp.stats.gamma.logpdf(x=y, a=nu, scale=mu / nu)

    @divide_weights
    def V(self, mu):
        """
        glm Variance function

        computes the variance of the distribution

        Parameters
        ----------
        mu : array-like of length n
            expected values

        Returns
        -------
        variance : np.array of length n
        """
        return mu**2

    @multiply_weights
    def deviance(self, y, mu, scaled=True):
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

        Returns
        -------
        deviances : np.array of length n
        """
        dev = 2 * ((y - mu) / mu - np.log(y / mu))

        if scaled:
            dev /= self.scale
        return dev

    def sample(self, mu):
        """
        Return random samples from this Gamma distribution.

        Parameters
        ----------
        mu : array-like of shape n_samples or shape (n_simulations, n_samples)
            expected values

        Returns
        -------
        random_samples : np.array of same shape as mu
        """
        # in numpy.random.gamma, `shape` is the parameter sometimes denoted by
        # `k` that corresponds to `nu` in S. Wood (2006) Table 2.1
        shape = 1.0 / self.scale
        # in numpy.random.gamma, `scale` is the parameter sometimes denoted by
        # `theta` that corresponds to mu / nu in S. Wood (2006) Table 2.1
        scale = mu / shape
        return np.random.gamma(shape=shape, scale=scale, size=None)


class InvGaussDist(Distribution):
    """
    Inverse Gaussian (Wald) Distribution
    """

    def __init__(self, scale=None):
        """
        creates an instance of the InvGaussDist class

        Parameters
        ----------
        scale : float or None, default: None
            scale/standard deviation of the distribution

        Returns
        -------
        self
        """
        super(InvGaussDist, self).__init__(name='inv_gauss', scale=scale)

    def log_pdf(self, y, mu, weights=None):
        """
        computes the log of the pdf or pmf of the values under the current distribution

        Parameters
        ----------
        y : array-like of length n
            target values
        mu : array-like of length n
            expected values
        weights : array-like shape (n,) or None, default: None
            containing sample weights
            if None, defaults to array of ones

        Returns
        -------
        pdf/pmf : np.array of length n
        """
        if weights is None:
            weights = np.ones_like(mu)
        gamma = weights / self.scale
        return sp.stats.invgauss.logpdf(y, mu, scale=1.0 / gamma)

    @divide_weights
    def V(self, mu):
        """
        glm Variance function

        computes the variance of the distribution

        Parameters
        ----------
        mu : array-like of length n
            expected values

        Returns
        -------
        variance : np.array of length n
        """
        return mu**3

    @multiply_weights
    def deviance(self, y, mu, scaled=True):
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

        Returns
        -------
        deviances : np.array of length n
        """
        dev = ((y - mu) ** 2) / (mu**2 * y)

        if scaled:
            dev /= self.scale
        return dev

    def sample(self, mu):
        """
        Return random samples from this Inverse Gaussian (Wald) distribution.

        Parameters
        ----------
        mu : array-like of shape n_samples or shape (n_simulations, n_samples)
            expected values

        Returns
        -------
        random_samples : np.array of same shape as mu
        """
        return np.random.wald(mean=mu, scale=self.scale, size=None)


import scipy.special as sp_special


class TweedieDist(Distribution):
    """
    Tweedie Distribution
    """

    def __init__(self, power, scale=None):
        if 0 < power < 1:
            raise ValueError(
                "Power parameter p must not be between 0 and 1 (non-inclusive) for the Tweedie distribution."
            )
        super(TweedieDist, self).__init__(name='tweedie', scale=scale)
        self.power = power

    def log_pdf(self, y, mu, weights=None):
        """
        Computes the log of the PDF (or PMF for y=0) of the Tweedie distribution.

        Parameters
        ----------
        y : array-like of length n
            Target values.
        mu : array-like of length n
            Expected values.
        weights : array-like shape (n,) or None, default: None
            Sample weights. If None, defaults to array of ones.

        Returns
        -------
        log_pdf : np.array of length n
        """
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)

        if np.any(y < 0) or np.any(mu <= 0):
            raise ValueError(
                "Negative values detected in y or non-positive values in mu."
            )

        if weights is None:
            weights = np.ones_like(mu)

        phi = self.scale  # dispersion parameter
        p = self.power  # power parameter

        # Initialize log_pdf array
        log_pdf_values = np.zeros_like(y, dtype=np.float64)

        # Indices where y > 0
        idx_positive = y > 0
        # Indices where y == 0
        idx_zero = y == 0

        # For y == 0
        if np.any(idx_zero):
            mu_zero = mu[idx_zero]
            # Compute log probability mass at zero
            log_p_zero = -(mu_zero ** (2 - p)) / ((2 - p) * phi)
            log_pdf_values[idx_zero] = log_p_zero

        # For y > 0
        if np.any(idx_positive):
            y_pos = y[idx_positive]
            mu_pos = mu[idx_positive]

            # Compute terms carefully to avoid numerical issues
            term1 = (y_pos * mu_pos ** (1 - p)) / (1 - p)
            term2 = mu_pos ** (2 - p) / (2 - p)
            term3 = y_pos ** (2 - p) / (2 - p)

            log_pdf = (1 / phi) * (term1 - term2 - term3)

            # Adjust with log(y) term and constants if necessary
            # Since constants cancel out in likelihood ratios, they are often omitted
            log_pdf_values[idx_positive] = log_pdf

        return log_pdf_values

    @divide_weights
    def V(self, mu):
        return mu**self.power

    @multiply_weights
    def deviance(self, y, mu, scaled=True):
        """
        Computes the deviance for the Tweedie distribution.

        Parameters
        ----------
        y : array-like of length n
            Target values.
        mu : array-like of length n
            Expected values.
        scaled : boolean, default: True
            Whether to divide the deviance by the distribution scale.

        Returns
        -------
        deviance : np.array of length n
        """
        if np.any(y < 0) or np.any(mu < 0):
            raise ValueError("Negative values detected in y or mu.")
        p = self.power
        if p < 0:
            dev = 2 * (
                np.power(np.maximum(y, 0), 2 - p) / ((1 - p) * (2 - p))
                - y * np.power(mu, 1 - p) / (1 - p)
                + np.power(mu, 2 - p) / (2 - p)
            )
        elif p == 0:
            dev = (y - mu) ** 2
        elif p == 1:
            dev = 2 * (y * np.log(y / mu) - (y - mu))
        elif p == 2:
            dev = 2 * (np.log(mu / y) + (y / mu) - 1)
        else:
            dev = 2 * (
                np.power(y, 2 - p) / ((1 - p) * (2 - p))
                - y * np.power(mu, 1 - p) / (1 - p)
                + np.power(mu, 2 - p) / (2 - p)
            )
        if scaled:
            dev /= self.scale
        return dev

    def sample(self, mu):
        """
        Generates random samples from the Tweedie distribution.

        Parameters
        ----------
        mu : array-like of shape n_samples
            Expected values.

        Returns
        -------
        random_samples : np.array of same shape as mu
        """
        p = self.power
        phi = self.scale

        if p <= 1:
            raise ValueError(
                "Power parameter p must be > 1 for the Tweedie distribution."
            )
        elif 1 < p < 2:
            # Compound Poisson-Gamma
            lambda_ = mu ** (2 - p) / (phi * (2 - p))
            num_events = np.random.poisson(lambda_)
            # Gamma parameters
            a = (2 - p) / (p - 1)
            s = phi * (p - 1) * mu ** (p - 1)
            samples = np.zeros_like(mu)
            positive_indices = num_events > 0
            samples[positive_indices] = np.array(
                [
                    np.sum(np.random.gamma(shape=a, scale=s[i], size=n))
                    for i, n in enumerate(num_events[positive_indices])
                ]
            )

            return samples
        elif p == 2:
            # Corrected Gamma distribution parameters
            shape = 1 / phi
            scale = phi * mu
            return np.random.gamma(shape=shape, scale=scale)

        else:
            raise NotImplementedError(
                "Sampling not implemented for p > 2 in TweedieDist."
            )


DISTRIBUTIONS = {
    'normal': NormalDist,
    'poisson': PoissonDist,
    'binomial': BinomialDist,
    'gamma': GammaDist,
    'inv_gauss': InvGaussDist,
    'tweedie': TweedieDist,
}
