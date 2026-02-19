import numpy as np

from pygam import GAM, NegativeBinomialGAM
from pygam.distributions import NegativeBinomialDist


def test_negative_binomial_gam_init():
    gam = NegativeBinomialGAM()
    assert isinstance(gam.distribution, NegativeBinomialDist)
    assert gam.link == "log"
    assert gam.alpha == 1.0


def test_negative_binomial_gam_init_alpha():
    gam = NegativeBinomialGAM(alpha=0.5)
    assert gam.distribution.alpha == 0.5
    assert gam.alpha == 0.5


def test_negative_binomial_gam_fit():
    # Generate data
    np.random.seed(42)
    n = 100
    X = np.linspace(0, 10, n)
    lp = 2 + 0.5 * X + np.sin(X)
    mu = np.exp(lp)
    alpha = 0.5
    # r = 1/alpha = 2
    r = 1.0 / alpha
    p = r / (r + mu)
    y = np.random.negative_binomial(n=r, p=p)

    # Fit GAM
    gam = NegativeBinomialGAM(alpha=alpha).fit(X[:, None], y)

    assert gam._is_fitted
    assert len(gam.coef_) > 0


def test_negative_binomial_gam_predict():
    np.random.seed(42)
    n = 50
    X = np.linspace(0, 10, n)
    lp = 1 + 0.2 * X
    mu = np.exp(lp)
    alpha = 1.0
    r = 1.0 / alpha
    p = r / (r + mu)
    y = np.random.negative_binomial(n=r, p=p)

    gam = NegativeBinomialGAM(alpha=alpha).fit(X[:, None], y)

    preds = gam.predict(X[:, None])
    assert preds.shape == (n,)
    assert np.all(preds >= 0)


def test_custom_gam_negative_binomial():
    # Test using generic GAM class
    np.random.seed(42)
    X = np.random.rand(50, 1)
    y = np.random.randint(0, 10, 50)  # Just some count data

    # Needs explicit distribution instance to set alpha if not default
    dist = NegativeBinomialDist(alpha=0.5)
    gam = GAM(distribution=dist, link="log").fit(X, y)
    assert gam._is_fitted
    assert isinstance(gam.distribution, NegativeBinomialDist)
    assert gam.distribution.alpha == 0.5


def test_negative_binomial_gam_summary():
    # Smoke test for summary
    np.random.seed(42)
    X = np.random.rand(20, 1)
    y = np.random.randint(0, 5, 20)
    gam = NegativeBinomialGAM().fit(X, y)
    gam.summary()
