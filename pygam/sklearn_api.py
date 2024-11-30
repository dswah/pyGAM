"""
sklearn_api.py

This module provides scikit-learn compatible classes for Generalized Additive Models (GAM) regressors and classifiers.
It integrates pygam's GAM capabilities with scikit-learn's estimator interface, enabling seamless use in machine learning pipelines.
"""

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from pygam import GAM
from pygam.terms import te, TermList  # Import te for interactions
import numpy as np

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
    terms : 'auto' or TermList, default='auto'
        The terms to include in the model.
    interactions : list of tuples, optional
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
        interactions=None,  # Added interactions parameter
        callbacks=['deviance', 'diffs'],
        fit_intercept=True,
        max_iter=100,
        tol=1e-4,
        verbose=False,
        **gam_params
    ):
        self.distribution = distribution
        self.link = link
        self.terms = TermList() if terms == 'auto' else terms
        self.interactions = interactions  # Store interactions
        self.callbacks = callbacks
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.gam_params = gam_params
        
        # ...existing code...
        if self.interactions:
            # Ensure terms is a list before appending
            if isinstance(self.terms, str):
                self.terms = [self.terms]
            elif not isinstance(self.terms, list):
                self.terms = list(self.terms)
            # Convert interaction tuples to te instances
            for interaction in self.interactions:
                self.terms.append(te(*interaction))
        
        if not isinstance(self.terms, TermList):
            self.terms = TermList(*self.terms)
    
        self.model_ = GAM(
            distribution=self.distribution,
            link=self.link,
            terms=self.terms,
            callbacks=self.callbacks,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            **self.gam_params
        )

    def fit(self, X, y):
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
    terms : 'auto' or TermList, default='auto'
        The terms to include in the model.
    interactions : list of tuples, optional
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
        interactions=None,  # Added interactions parameter
        callbacks=['deviance', 'diffs', 'accuracy'],
        fit_intercept=True,
        max_iter=100,
        tol=1e-4,
        verbose=False,
        **gam_params
    ):
        self.distribution = distribution
        self.link = link
        self.terms = TermList() if terms == 'auto' else terms
        self.interactions = interactions  # Store interactions
        self.callbacks = callbacks
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.gam_params = gam_params
        
        # ...existing code...
        if self.interactions:
            # Ensure terms is a list before appending
            if isinstance(self.terms, str):
                self.terms = [self.terms]
            elif not isinstance(self.terms, list):
                self.terms = list(self.terms)
            # Convert interaction tuples to te instances
            for interaction in self.interactions:
                self.terms.append(te(*interaction))
        
        if not isinstance(self.terms, TermList):
            self.terms = TermList(*self.terms)
    
        self.model_ = GAM(
            distribution=self.distribution,
            link=self.link,
            terms=self.terms,
            callbacks=self.callbacks,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            **self.gam_params
        )

    def fit(self, X, y):
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
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))