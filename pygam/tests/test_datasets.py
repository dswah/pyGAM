# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pygam.datasets import cake
from pygam.datasets import coal
from pygam.datasets import default
from pygam.datasets import faithful
from pygam.datasets import hepatitis
from pygam.datasets import mcycle
from pygam.datasets import trees
from pygam.datasets import wage
from pygam.datasets import chicago
from pygam.datasets import toy_interaction

from pygam.datasets import __all__ as DATASETS

def _test_dataset(dataset_loader, n_rows, n_columns_X, n_columns_df, n_rows_X=None):
    """check the length of the dataset is the same regardless of the transformation
    check the columns of the dataset are correct in X_y and as a DataFrame

    check the transformation is correct

    check dtype is float for X_y

    check ndim for X is 2

    Parameters
    ----------
    dataset_loader : function, returns a dataframe or a tuple of arrays

    n_rows : int, expected number of rows in dataset

    n_columns_X : int, expected number of columns in the transformed
        dataset independent variables

    n_columns_df : int, expected number of columns in the original
        dataset dataframe

    n_rows_X : None, or int, expected number of rows in the transformed
        dataset independent variables if different from the original.
        This is usually necessary for datasets that use histogram transforms

    Returns
    -------
    None
    """
    if n_rows_X is None:
        n_rows_X = n_rows

    df = dataset_loader(return_X_y=False)
    X_y = dataset_loader(return_X_y=True)

    # number of rows never changes
    assert df.shape[0] == n_rows
    assert X_y[0].shape[0] == X_y[1].shape[0] == n_rows_X

    # check columns
    assert df.shape[1] == n_columns_df
    assert X_y[0].shape[1] == n_columns_X

    # check dtype
    assert X_y[0].dtype == X_y[1].dtype == 'float'

    # check shape
    assert X_y[0].ndim == 2

def test_cake():
    _test_dataset(cake, n_rows=270, n_columns_X=3, n_columns_df=5)

def test_coal():
    _test_dataset(coal, n_rows=191, n_columns_X=1, n_columns_df=1, n_rows_X=150)

def test_default():
    _test_dataset(default, n_rows=10000, n_columns_X=3, n_columns_df=4)

def test_faithful():
    _test_dataset(faithful, n_rows=272, n_columns_X=1, n_columns_df=2, n_rows_X=200)

def test_hepatitis():
    _test_dataset(hepatitis, n_rows=86, n_columns_X=1, n_columns_df=3, n_rows_X=83)

def test_mcycle():
    _test_dataset(mcycle, n_rows=133, n_columns_X=1, n_columns_df=2)

def test_trees():
    _test_dataset(trees, n_rows=31, n_columns_X=2, n_columns_df=3)

def test_chicago():
    _test_dataset(chicago, n_rows=5114, n_columns_X=4, n_columns_df=7, n_rows_X=4863)

def test_toy_interaction():
    _test_dataset(toy_interaction, n_rows=50000, n_columns_X=2, n_columns_df=3)
