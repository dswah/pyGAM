.. pyGAM documentation master file, created by
   sphinx-quickstart on Sat Aug 18 15:42:53 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyGAM's documentation!
=================================

pyGAM is a package for building Generalized Linear Models in Python,
whose structure will be immediately familiar to anyone with experience
of scipy or scikit-learn.

Installation
------------

pyGAM is on pypi, and can be installed using ``pip``:

``pip install pygam``

To speed up optimization on large models with constraints, it helps to
have ``scikit-sparse`` installed because it contains a slightly faster,
sparse version of Cholesky factorization. The import from
``scikit-sparse`` references ``nose``, so you'll need that too.

The easiest way is to use Conda:
``conda install -c conda-forge scikit-sparse``

`scikit-sparse docs
<http://pythonhosted.org/scikit-sparse/overview.html#download>`_

If you're new to pyGAM, read :ref:`the Getting Started guide </notebooks/getting_started.ipynb>`
for an introduction to the package.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    notebooks/getting_started.ipynb
    api/pygam.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
