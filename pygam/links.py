"""Link Functions"""

import numpy as np

from pygam.core import Core


class Link(Core):
    """
    Creates an instance of a Link object.

    Parameters
    ----------
    name : str, default: None
    """

    def __init__(self, name=None):
        super(Link, self).__init__(name=name)


class IdentityLink(Link):
    """
    Identity Link

    Parameters
    ----------
    """

    def __init__(self):
        super(IdentityLink, self).__init__(name="identity")

    def link(self, mu, dist):
        """
        Glm link function
        this is useful for going from mu to the linear prediction.

        Parameters
        ----------
        mu : array-like of length n
        dist : Distribution instance

        Returns
        -------
        lp : np.array of length n
        """
        return mu

    def mu(self, lp, dist):
        """
        Glm mean function, ie inverse of link function
        this is useful for going from the linear prediction to mu.

        Parameters
        ----------
        lp : array-like of length n
        dist : Distribution instance

        Returns
        -------
        mu : np.array of length n
        """
        return lp

    def gradient(self, mu, dist):
        """
        Derivative of the link function wrt mu.

        Parameters
        ----------
        mu : array-like of length n
        dist : Distribution instance

        Returns
        -------
        grad : np.array of length n
        """
        return np.ones_like(mu)

    def g_double_prime(self, mu, dist):
        """Second derivative of link wrt mu."""
        return np.zeros_like(mu)

    def g_triple_prime(self, mu, dist):
        """Third derivative of link wrt mu."""
        return np.zeros_like(mu)


class LogitLink(Link):
    """
    Logit Link

    Parameters
    ----------
    """

    def __init__(self):
        super(LogitLink, self).__init__(name="logit")

    def link(self, mu, dist):
        """
        Glm link function
        this is useful for going from mu to the linear prediction.

        Parameters
        ----------
        mu : array-like of length n
        dist : Distribution instance

        Returns
        -------
        lp : np.array of length n
        """
        return np.log(mu) - np.log(dist.levels - mu)

    def mu(self, lp, dist):
        """
        Glm mean function, ie inverse of link function
        this is useful for going from the linear prediction to mu.

        Parameters
        ----------
        lp : array-like of length n
        dist : Distribution instance

        Returns
        -------
        mu : np.array of length n
        """
        elp = np.exp(lp)
        return dist.levels * elp / (elp + 1)

    def gradient(self, mu, dist):
        """
        Derivative of the link function wrt mu.

        Parameters
        ----------
        mu : array-like of length n
        dist : Distribution instance

        Returns
        -------
        grad : np.array of length n
        """
        return dist.levels / (mu * (dist.levels - mu))

    def g_double_prime(self, mu, dist):
        """Second derivative of link wrt mu."""
        L = dist.levels
        denom = (mu * (L - mu)) ** 2
        return -L * (L - 2 * mu) / denom

    def g_triple_prime(self, mu, dist):
        """Third derivative of link wrt mu computed analytically."""
        L = dist.levels
        denom = mu * (L - mu)
        # g'' = -L*(L-2mu)/(denom^2)
        # differentiate using quotient/product rules
        term1 = 2 * L * (L - 2 * mu) ** 2 / (denom**3)
        term2 = 2 * L / (denom**2)
        return term1 + term2


class LogLink(Link):
    """
    Log Link

    Parameters
    ----------
    """

    def __init__(self):
        super(LogLink, self).__init__(name="log")

    def link(self, mu, dist):
        """
        Glm link function
        this is useful for going from mu to the linear prediction.

        Parameters
        ----------
        mu : array-like of length n
        dist : Distribution instance

        Returns
        -------
        lp : np.array of length n
        """
        return np.log(mu)

    def mu(self, lp, dist):
        """
        Glm mean function, ie inverse of link function
        this is useful for going from the linear prediction to mu.

        Parameters
        ----------
        lp : array-like of length n
        dist : Distribution instance

        Returns
        -------
        mu : np.array of length n
        """
        return np.exp(lp)

    def gradient(self, mu, dist):
        """
        Derivative of the link function wrt mu.

        Parameters
        ----------
        mu : array-like of length n
        dist : Distribution instance

        Returns
        -------
        grad : np.array of length n
        """
        return 1.0 / mu

    def g_double_prime(self, mu, dist):
        """Second derivative of link wrt mu."""
        return -1.0 / (mu**2)

    def g_triple_prime(self, mu, dist):
        """Third derivative of link wrt mu."""
        return 2.0 / (mu**3)


class InverseLink(Link):
    """
    Inverse Link

    Parameters
    ----------
    """

    def __init__(self):
        super(InverseLink, self).__init__(name="inverse")

    def link(self, mu, dist):
        """
        Glm link function
        this is useful for going from mu to the linear prediction.

        Parameters
        ----------
        mu : array-like of length n
        dist : Distribution instance

        Returns
        -------
        lp : np.array of length n
        """
        return mu**-1.0

    def mu(self, lp, dist):
        """
        Glm mean function, ie inverse of link function
        this is useful for going from the linear prediction to mu.

        Parameters
        ----------
        lp : array-like of length n
        dist : Distribution instance

        Returns
        -------
        mu : np.array of length n
        """
        return lp**-1.0

    def gradient(self, mu, dist):
        """
        Derivative of the link function wrt mu.

        Parameters
        ----------
        mu : array-like of length n
        dist : Distribution instance

        Returns
        -------
        grad : np.array of length n
        """
        return -1 * mu**-2.0

    def g_double_prime(self, mu, dist):
        return 2.0 * mu**-3.0

    def g_triple_prime(self, mu, dist):
        return -6.0 * mu**-4.0


class InvSquaredLink(Link):
    """
    Inverse Squared Link

    Parameters
    ----------
    """

    def __init__(self):
        super(InvSquaredLink, self).__init__(name="inv_squared")

    def link(self, mu, dist):
        """
        Glm link function
        this is useful for going from mu to the linear prediction.

        Parameters
        ----------
        mu : array-like of length n
        dist : Distribution instance

        Returns
        -------
        lp : np.array of length n
        """
        return mu**-2.0

    def mu(self, lp, dist):
        """
        Glm mean function, ie inverse of link function
        this is useful for going from the linear prediction to mu.

        Parameters
        ----------
        lp : array-like of length n
        dist : Distribution instance

        Returns
        -------
        mu : np.array of length n
        """
        return lp**-0.5

    def gradient(self, mu, dist):
        """
        Derivative of the link function wrt mu.

        Parameters
        ----------
        mu : array-like of length n
        dist : Distribution instance

        Returns
        -------
        grad : np.array of length n
        """
        return -2 * mu**-3.0

    def g_double_prime(self, mu, dist):
        return 6.0 * mu**-4.0

    def g_triple_prime(self, mu, dist):
        return -24.0 * mu**-5.0


LINKS = {
    "identity": IdentityLink,
    "log": LogLink,
    "logit": LogitLink,
    "inverse": InverseLink,
    "inv_squared": InvSquaredLink,
}
