import pytest
import numpy as np
from pygam.distributions import TweedieDist
from pygam import GAM, s

@pytest.fixture
def tweedie_dist():
    return TweedieDist(power=1.5, scale=1.0)

def test_log_pdf(tweedie_dist):
    mu = np.array([1.0, 2.0, 3.0])
    y = np.array([1.5, 2.5, 3.5])
    weights = np.array([1.0, 1.0, 1.0])
    log_pdf = tweedie_dist.log_pdf(y, mu, weights)
    assert log_pdf.shape == y.shape
    assert np.all(np.isfinite(log_pdf)), "Log PDF contains non-finite values."

def test_deviance(tweedie_dist):
    mu = np.array([1.0, 2.0, 3.0])
    y = np.array([1.5, 2.5, 3.5])
    deviance = tweedie_dist.deviance(y, mu, scaled=True)
    assert deviance.shape == y.shape
    assert np.all(deviance >= 0), "Deviance contains negative values."

def test_sample(tweedie_dist):
    mu = np.array([1.0, 2.0, 3.0])
    # Generate 1000 samples for each mu
    samples = np.array([tweedie_dist.sample(mu) for _ in range(100)])
    sample_mean = np.mean(samples)
    expected_mean = np.mean(mu)
    # Adjust the tolerance based on the variance
    tolerance = 0.1 * expected_mean
    assert abs(sample_mean - expected_mean) < tolerance, "Sample mean is not within the expected range."


def test_invalid_power():
    with pytest.raises(ValueError):
        TweedieDist(power=0.5, scale=1.0)  # Power less than 1 is invalid

def test_not_implemented_power():
    dist = TweedieDist(power=3.0, scale=1.0)
    mu = np.array([1.0, 2.0, 3.0])
    with pytest.raises(NotImplementedError):
        dist.sample(mu)

def test_gam_tweedie_fit():
    # Generate synthetic data
    np.random.seed(0)
    n_samples = 100
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    true_coef = 2.0
    # Generate target variable following a Tweedie distribution
    # For simplicity, using Gamma distribution as a proxy when power=2
    y = X.flatten() * true_coef + np.random.gamma(shape=2.0, scale=1.0, size=n_samples)

    # Initialize and fit GAM with Tweedie distribution
    gam = GAM(terms=s(0), distribution='tweedie', power=1.5, fit_intercept=True)
    gam.fit(X, y)

    # Predict and check the shape of predictions
    y_pred = gam.predict(X)
    assert y_pred.shape == y.shape, "Predictions shape mismatch."

    # Check that predictions are finite
    assert np.all(np.isfinite(y_pred)), "Predictions contain non-finite values."

    # Optionally, check if the mean of predictions is close to the mean of y
    sample_mean = np.mean(y_pred)
    expected_mean = np.mean(y)
    assert abs(sample_mean - expected_mean) < 1.0, "Sample mean is not within the expected range."

def test_variance_function(tweedie_dist):
    mu = np.array([1.0, 2.0, 3.0])
    variance = tweedie_dist.V(mu)
    expected_variance = mu ** tweedie_dist.power
    assert np.allclose(variance, expected_variance), "Variance function V(mu) is incorrect."

def test_zero_targets(tweedie_dist):
    mu = np.array([1.0, 2.0, 3.0])
    y = np.array([0.0, 0.0, 0.0])
    log_pdf = tweedie_dist.log_pdf(y, mu)
    assert log_pdf.shape == y.shape
    assert np.all(np.isfinite(log_pdf)), "Log PDF with zero targets contains non-finite values."

def test_negative_inputs(tweedie_dist):
    mu = np.array([-1.0, 2.0, 3.0])
    y = np.array([1.0, -2.0, 3.0])
    with pytest.raises(ValueError):
        tweedie_dist.log_pdf(y, mu)
    with pytest.raises(ValueError):
        tweedie_dist.deviance(y, mu)

def test_sample_with_zero_mu(tweedie_dist):
    mu = np.array([0.0, 0.0, 0.0])
    samples = tweedie_dist.sample(mu)
    assert np.all(samples == 0), "Samples with zero mu should be zeros."

def test_boundary_power_values():
    mu = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])

    # Power approaching 1
    tweedie_dist = TweedieDist(power=1.0001, scale=1.0)
    log_pdf = tweedie_dist.log_pdf(y, mu)
    assert np.all(np.isfinite(log_pdf)), "Log PDF near power=1 contains non-finite values."

    # Power approaching 2
    tweedie_dist = TweedieDist(power=1.9999, scale=1.0)
    log_pdf = tweedie_dist.log_pdf(y, mu)
    assert np.all(np.isfinite(log_pdf)), "Log PDF near power=2 contains non-finite values."
