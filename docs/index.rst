.. pyGAM documentation master file, created by
   sphinx-quickstart on Sat Aug 18 15:42:53 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to pyGAM's documentation!
=================================

.. image:: ../imgs/pygam_tensor.png
    :width: 450px
    :alt: pyGAM logo
    :align: center

|Build Status| |Documentation Status| |Coverage| |PyPi Version| |Python Versions| |Zenodo| |Open Source|

|

Getting Started
===============

pyGAM is a package for building Generalized Additive Models in Python,
with an emphasis on modularity and performance. The API will be immediately familiar to anyone with experience
of scikit-learn or scipy.

If you're new to pyGAM, take a :ref:`Tour of pyGAM </notebooks/tour_of_pygam.ipynb>`
for an introduction to the package.

|

Installation
============

Pip
---
pyGAM is on pypi, and can be installed using ``pip``: ::

  pip install pygam

Conda
-----
Or via ``conda-forge``, however this is typically less up-to-date: ::

  conda install -c conda-forge pyGAM

Bleeding Edge
-------------
You can install the bleeding edge from github using ``pip``.
First clone the repo, ``cd`` into the main directory and do: ::

  pip install .  # for an unstable "latest" dev version
  # or
  pip install -e .  # for an editable developer/contributor

Acceleration
------------
Most of pyGAM's computations are linear algebra operations.

To speed up optimization on large models with constraints, it helps to have `intel MKL <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html>`_ installed.

It is currently a bit tricky to install both NumPy and SciPy linked to the MKL routines with Conda because you have to be careful with which channel you are using. Pip's NumPy-MKL is outdated.

An alternative is to use a third-party build like https://urob.github.io/numpy-mkl: ::

  pip install numpy scipy --extra-index-url https://urob.github.io/numpy-mkl

|

Dependencies
============
pyGAM is tested on Python 3.10+ and depends on ``NumPy``, ``SciPy``, and ``progressbar2``.

In addition to the above dependencies, the ``pygam.datasets`` submodule relies on ``Pandas``.

See `pyproject.toml <https://github.com/dswah/pyGAM/blob/main/pyproject.toml>`_ for detailed version information).

|

Citing pyGAM
============

  Servén D., Brummitt C. (2018). pyGAM: Generalized Additive Models in Python. Zenodo. `DOI: 10.5281/zenodo.1208723 <http://doi.org/10.5281/zenodo.1208723>`_

|

Contact
=======
To report an issue with pyGAM please use the `issue tracker <https://github.com/dswah/pyGAM/issues>`_.

|

License
=======
`Apache Software License 2.0 <https://github.com/dswah/pyGAM/blob/main/LICENSE>`_

|

Quick Start
===========
.. toctree::
    :maxdepth: 2

    notebooks/quick_start.ipynb

|

Tour of pyGAM
=============
.. toctree::
    :maxdepth: 2

    notebooks/tour_of_pygam.ipynb

|

User API
========
.. toctree::
   :maxdepth: 2

   reference/index

|

Indices and tables
==================
:ref:`genindex`



.. |Build Status| image:: https://img.shields.io/github/actions/workflow/status/dswah/pygam/pypi.yml?logo=github
   :target: https://github.com/dswah/pygam/actions/workflows/pypi.yml
.. |Documentation Status| image:: https://img.shields.io/readthedocs/pygam?logo=readthedocs
   :target: https://pygam.readthedocs.io/en/latest/?badge=latest
.. |Coverage| image:: https://codecov.io/gh/dswah/pygam/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/dswah/pygam
.. |PyPi Version| image:: https://badge.fury.io/py/pygam.svg
   :target: https://badge.fury.io/py/pygam
.. |Python Versions| image:: https://shields.io/badge/python-3.10+-blue
.. |Zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1208723.svg
   :target: https://doi.org/10.5281/zenodo.1208723
.. |Open Source| image:: https://img.shields.io/badge/powered%20by-Open%20Source-orange.svg?style=flat&colorA=E1523D&colorB=007D8A
   :target: https://github.com/dswah/pyGAM
