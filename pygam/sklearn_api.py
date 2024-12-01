"""
sklearn_api.py

This module provides scikit-learn compatible classes for Generalized Additive Models (GAM) regressors and classifiers.
It integrates pygam's GAM capabilities with scikit-learn's estimator interface, enabling seamless use in machine learning pipelines.
"""

# Standard library imports
import numpy as np

# Third-party imports
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Local application imports
from pygam import GAM
from pygam.terms import te, TermList, Term  # Import te for interactions
from pygam.terms import s, f, l, intercept  # Import s, f, l for splines


def create_default_terms(X, categorical_features=None):
    """Generate default terms for each feature in X, handling categoricals."""
    n_features = X.shape[1]
    terms = []
    if categorical_features is None:
        categorical_features = []
    for i in range(n_features):
        if i in categorical_features:
            terms.append(f(i))
        elif not np.issubdtype(X[:, i].dtype, np.number):
            terms.append(f(i))
        else:
            terms.append(s(i))
    return terms


class GAMRegressor(BaseEstimator, RegressorMixin):
    """
    GAMRegressor

    A scikit-learn compatible regressor using Generalized Additive Models (GAM).

    Parameters
    ----------
    distribution : str, default='normal'
        The distribution of the response variable.
    link : str, default='identity'
        The link function.
    terms : 'auto', None, or list of Term objects, default='auto'
        The terms to include in the model. If 'auto', terms are automatically inferred based on X.
        If None, no terms are used. If a list of Term objects, they are used as specified.
    interactions : None or list of Term objects, optional
        Interaction terms to include in the model.
    callbacks : list, default=['deviance', 'diffs']
        List of callbacks to monitor during training.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    verbose : bool, default=False
        Verbosity mode.
    categorical_features : list, optional
        List of indices of categorical features.
    **gam_params :
        Additional parameters for the GAM model.

    Attributes
    ----------
    model_ : GAM
        The underlying pygam GAM model fitted to the data.
    """

    def __init__(
        self,
        distribution='normal',
        link='identity',
        terms='auto',
        interactions=None,
        callbacks=['deviance', 'diffs'],
        fit_intercept=True,
        max_iter=100,
        tol=1e-4,
        verbose=False,
        categorical_features=None,
        **gam_params,
    ):
        self.distribution = distribution
        self.link = link
        self.terms = terms
        self.interactions = interactions
        self.callbacks = callbacks
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.categorical_features = categorical_features
        self.gam_params = gam_params

    def fit(self, X, y):
        if self.terms == 'auto':
            self.terms_ = create_default_terms(X, self.categorical_features)
        elif self.terms is None:
            self.terms_ = []
        else:
            self.terms_ = self.terms

        if self.interactions is not None:
            self.interactions_ = [
                te(*interaction) if isinstance(interaction, tuple) else interaction
                for interaction in self.interactions
            ]
        else:
            self.interactions_ = []

        # Combine terms and interactions
        terms = self.terms_ + self.interactions_
        terms = TermList(*terms)

        # Create the GAM model with the specified terms
        self.model_ = GAM(
            distribution=self.distribution,
            link=self.link,
            terms=terms,
            callbacks=self.callbacks,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            **self.gam_params,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def score(self, X, y):
        return float(self.model_.statistics_.get('pseudo R-squared', 0))


class GAMClassifier(BaseEstimator, ClassifierMixin):
    """
    GAMClassifier

    A scikit-learn compatible classifier using Generalized Additive Models (GAM).

    Parameters
    ----------
    distribution : str, default='binomial'
        The distribution of the response variable.
    link : str, default='logit'
        The link function.
    terms : 'auto', None, or list of Term objects, default='auto'
        The terms to include in the model. If 'auto', terms are automatically inferred based on X.
        If None, no terms are used. If a list of Term objects, they are used as specified.
    interactions : None or list of Term objects, optional
        Interaction terms to include in the model.
    callbacks : list, default=['deviance', 'diffs', 'accuracy']
        List of callbacks to monitor during training.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    verbose : bool, default=False
        Verbosity mode.
    categorical_features : list, optional
        List of indices of categorical features.
    **gam_params :
        Additional parameters for the GAM model.

    Attributes
    ----------
    model_ : GAM
        The underlying pygam GAM model fitted to the data.
    classes_ : array-like
        Unique class labels.
    """

    def __init__(
        self,
        distribution='binomial',
        link='logit',
        terms='auto',
        interactions=None,
        callbacks=['deviance', 'diffs', 'accuracy'],
        fit_intercept=True,
        max_iter=100,
        tol=1e-4,
        verbose=False,
        categorical_features=None,
        **gam_params,
    ):
        self.distribution = distribution
        self.link = link
        self.terms = terms
        self.interactions = interactions
        self.callbacks = callbacks
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.categorical_features = categorical_features
        self.gam_params = gam_params

    def fit(self, X, y):
        if self.terms == 'auto':
            self.terms_ = create_default_terms(X, self.categorical_features)
        elif self.terms is None:
            self.terms_ = []
        else:
            self.terms_ = self.terms

        if self.interactions is not None:
            self.interactions_ = [
                te(*interaction) if isinstance(interaction, tuple) else interaction
                for interaction in self.interactions
            ]
        else:
            self.interactions_ = []

        # Combine terms and interactions
        terms = self.terms_ + self.interactions_
        terms = TermList(*terms)

        # Create the GAM model with the specified terms
        self.model_ = GAM(
            distribution=self.distribution,
            link=self.link,
            terms=terms,
            callbacks=self.callbacks,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            **self.gam_params,
        )
        self.model_.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        proba = self.model_.predict(X)
        if len(self.classes_) == 2:
            return (proba >= 0.5).astype(int)
        else:
            return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        proba = self.model_.predict(X)
        if len(self.classes_) == 2:
            return np.vstack([1 - proba, proba]).T
        else:
            return proba  # Assume GAM model returns probabilities for each class

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
