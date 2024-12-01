import pytest
import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from pygam.sklearn_api import GAMRegressor, GAMClassifier

@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def test_gam_regressor_fit_predict(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    reg = GAMRegressor()
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)
    assert predictions.shape == y_test.shape
    assert r2_score(y_test, predictions) >= 0  # Basic sanity check

def test_gam_regressor_score(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    reg = GAMRegressor()
    reg.fit(X_train, y_train)
    score = reg.score(X_test, y_test)
    assert isinstance(score, float)
    assert score >= 0  # R-squared should be non-negative

def test_gam_classifier_fit_predict(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    clf = GAMClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    assert predictions.shape == y_test.shape
    assert set(predictions).issubset({0, 1})  # Binary classification

def test_gam_classifier_predict_proba(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    clf = GAMClassifier()
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    assert proba.shape == (X_test.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1)

def test_gam_classifier_score(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    clf = GAMClassifier()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    assert isinstance(score, float)
    assert 0 <= score <= 1  # Accuracy between 0 and 1

def test_gam_regressor_with_custom_params(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    reg = GAMRegressor(distribution='normal', link='identity', max_iter=200, tol=1e-5)
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)
    assert r2_score(y_test, predictions) >= 0

def test_gam_classifier_with_custom_params(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    clf = GAMClassifier(distribution='binomial', link='logit', max_iter=200, tol=1e-5)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    assert accuracy_score(y_test, predictions) >= 0
    assert proba.shape == (X_test.shape[0], 2)

def test_gam_regressor_with_callbacks(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    reg = GAMRegressor(callbacks=['deviance', 'diffs'])
    reg.fit(X_train, y_train)
    assert 'deviance' in reg.model_.logs_
    assert 'diffs' in reg.model_.logs_

def test_gam_classifier_with_callbacks(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    clf = GAMClassifier(callbacks=['deviance', 'diffs', 'accuracy'])
    clf.fit(X_train, y_train)
    assert 'deviance' in clf.model_.logs_
    assert 'diffs' in clf.model_.logs_
    assert 'accuracy' in clf.model_.logs_

def test_gam_regressor_gamma():
    X = np.random.rand(100, 2)
    y = np.random.gamma(shape=2.0, scale=1.0, size=100)
    model = GAMRegressor(distribution='gamma')
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == y.shape

def test_gam_regressor_poisson():
    X = np.random.rand(100, 2)
    y = np.random.poisson(lam=3.0, size=100)
    model = GAMRegressor(distribution='poisson')
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == y.shape

def test_gam_regressor_with_interactions(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    interactions = [(0, 1), (2, 3)]  # Specify feature indices for interactions
    reg = GAMRegressor(interactions=interactions)
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)
    assert predictions.shape == y_test.shape
    assert r2_score(y_test, predictions) >= 0  # Basic sanity check

def test_gam_classifier_with_interactions(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    interactions = [(0, 1), (2, 3)]  # Specify feature indices for interactions
    clf = GAMClassifier(interactions=interactions)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    assert predictions.shape == y_test.shape
    assert set(predictions).issubset({0, 1})  # Binary classification
    assert proba.shape == (X_test.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1)
    assert accuracy_score(y_test, predictions) >= 0
