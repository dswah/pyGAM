"""Link Functions"""

import numpy as np
from scipy import stats

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


class SqrtLink(Link):
    """
    Square-root Link

    Parameters
    ----------
    """

    def __init__(self):
        super(SqrtLink, self).__init__(name="sqrt")

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
        return np.sqrt(mu)

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
        return lp**2.0

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
        return 0.5 * mu**-0.5


class ProbitLink(Link):
    """
    Probit Link

    Parameters
    ----------
    """

    def __init__(self):
        super(ProbitLink, self).__init__(name="probit")

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
        return stats.norm.ppf(mu)

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
        return stats.norm.cdf(lp)

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
        z = stats.norm.ppf(mu)
        return 1.0 / stats.norm.pdf(z)


class CLogLogLink(Link):
    """
    Complementary Log-Log Link

    Parameters
    ----------
    """

    def __init__(self):
        super(CLogLogLink, self).__init__(name="cloglog")

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
        return np.log(-np.log(1.0 - mu))

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
        return 1.0 - np.exp(-np.exp(lp))

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
        return 1.0 / ((1.0 - mu) * -np.log(1.0 - mu))


class CauchitLink(Link):
    """
    Cauchit Link

    Parameters
    ----------
    """

    def __init__(self):
        super(CauchitLink, self).__init__(name="cauchit")

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
        return np.tan(np.pi * (mu - 0.5))

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
        return 0.5 + np.arctan(lp) / np.pi

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
        return np.pi * (1.0 + np.tan(np.pi * (mu - 0.5)) ** 2)


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


LINKS = {
    "identity": IdentityLink,
    "log": LogLink,
    "logit": LogitLink,
    "inverse": InverseLink,
    "inv_squared": InvSquaredLink,
    "sqrt": SqrtLink,
    "probit": ProbitLink,
    "cloglog": CLogLogLink,
    "cauchit": CauchitLink,
}
