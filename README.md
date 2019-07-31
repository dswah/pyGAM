[![Build Status](https://travis-ci.org/dswah/pyGAM.svg?branch=master)](https://travis-ci.org/dswah/pyGAM)
[![Documentation Status](https://readthedocs.org/projects/pygam/badge/?version=latest)](https://pygam.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/pygam.svg)](https://badge.fury.io/py/pygam)
[![codecov](https://codecov.io/gh/dswah/pygam/branch/master/graph/badge.svg)](https://codecov.io/gh/dswah/pygam)
[![python27](https://img.shields.io/badge/python-2.7-blue.svg)](https://badge.fury.io/py/pygam)
[![python36](https://img.shields.io/badge/python-3.6-blue.svg)](https://badge.fury.io/py/pygam)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1208723.svg)](https://doi.org/10.5281/zenodo.1208723)


# pyGAM
Generalized Additive Models in Python.

<img src=imgs/pygam_tensor.png>

## Documentation
- [Official pyGAM Documentation: Read the Docs](https://pygam.readthedocs.io/en/latest/?badge=latest)  
- [Building interpretable models with Generalized additive models in Python](https://medium.com/just-another-data-scientist/building-interpretable-models-with-generalized-additive-models-in-python-c4404eaf5515)  
<!-----
[pyGAM: Getting started with Generalized Additive Models in Python](https://medium.com/@jpoberhauser/pygam-getting-started-with-generalized-additive-models-in-python-457df5b4705f)
----->

## Installation
```pip install pygam```

### scikit-sparse
To speed up optimization on large models with constraints, it helps to have `scikit-sparse` installed because it contains a slightly faster, sparse version of Cholesky factorization. The import from `scikit-sparse` references `nose`, so you'll need that too.

The easiest way is to use Conda:  
```conda install -c conda-forge scikit-sparse nose```

[scikit-sparse docs](http://pythonhosted.org/scikit-sparse/overview.html#download)

## Contributing - HELP REQUESTED
Contributions are most welcome!

You can help pyGAM in many ways including:

- Working on a [known bug](https://github.com/dswah/pyGAM/labels/bug).
- Trying it out and reporting bugs or what was difficult.
- Helping improve the documentation.
- Writing new [distributions](https://github.com/dswah/pyGAM/blob/master/pygam/distributions.py), and [link functions](https://github.com/dswah/pyGAM/blob/master/pygam/links.py).
- If you need some ideas, please take a look at the [issues](https://github.com/dswah/pyGAM/issues).


To start:
- **fork the project** and cut a new branch
- Now **install** the testing **dependencies**

```
conda install pytest numpy pandas scipy pytest-cov cython
pip install --upgrade pip
pip install -r requirements.txt
```

It helps to add a **sym-link** of the forked project to your **python path**. To do this, you should **install [flit](http://flit.readthedocs.io/en/latest/index.html)**:
- ```pip install flit```
- Then from main project folder (ie `.../pyGAM`) do:
```flit install -s```

Make some changes and write a test...
- **Test** your contribution (eg from the `.../pyGAM`):
```py.test -s```
- When you are happy with your changes, make a **pull request** into the `master` branch of the main project.


## About
Generalized Additive Models (GAMs) are smooth semi-parametric models of the form:

![alt tag](http://latex.codecogs.com/svg.latex?g\(\mathbb{E}\[y|X\]\)=\beta_0+f_1(X_1)+f_2(X_2)+\dots+f_p(X_p))

where `X.T = [X_1, X_2, ..., X_p]` are independent variables, `y` is the dependent variable, and `g()` is the link function that relates our predictor variables to the expected value of the dependent variable.

The feature functions `f_i()` are built using **penalized B splines**, which allow us to **automatically model non-linear relationships** without having to manually try out many different transformations on each variable.

<img src=imgs/pygam_basis.png>

GAMs extend generalized linear models by allowing non-linear functions of features while maintaining additivity. Since the model is additive, it is easy to examine the effect of each `X_i` on `Y` individually while holding all other predictors constant.

The result is a very flexible model, where it is easy to incorporate prior knowledge and control overfitting.

## Citing pyGAM
Please consider citing pyGAM if it has helped you in your research or work:

Daniel Servén, & Charlie Brummitt. (2018, March 27). pyGAM: Generalized Additive Models in Python. Zenodo. [DOI: 10.5281/zenodo.1208723](http://doi.org/10.5281/zenodo.1208723)

BibTex:
```
@misc{daniel\_serven\_2018_1208723,
  author       = {Daniel Servén and
                  Charlie Brummitt},
  title        = {pyGAM: Generalized Additive Models in Python},
  month        = mar,
  year         = 2018,
  doi          = {10.5281/zenodo.1208723},
  url          = {https://doi.org/10.5281/zenodo.1208723}
}
```

## References
1. Simon N. Wood, 2006  
Generalized Additive Models: an introduction with R

0. Hastie, Tibshirani, Friedman  
The Elements of Statistical Learning  
http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf  

0. James, Witten, Hastie and Tibshirani  
An Introduction to Statistical Learning  
http://www-bcf.usc.edu/~gareth/ISL/ISLR%20Sixth%20Printing.pdf  

0. Paul Eilers & Brian Marx, 1996
Flexible Smoothing with B-splines and Penalties
http://www.stat.washington.edu/courses/stat527/s13/readings/EilersMarx_StatSci_1996.pdf

0. Kim Larsen, 2015  
GAM: The Predictive Modeling Silver Bullet  
http://multithreaded.stitchfix.com/assets/files/gam.pdf  

0. Deva Ramanan, 2008  
UCI Machine Learning: Notes on IRLS  
http://www.ics.uci.edu/~dramanan/teaching/ics273a_winter08/homework/irls_notes.pdf  

0. Paul Eilers & Brian Marx, 2015  
International Biometric Society: A Crash Course on P-splines  
http://www.ibschannel2015.nl/project/userfiles/Crash_course_handout.pdf

0. Keiding, Niels, 1991  
Age-specific incidence and prevalence: a statistical perspective


<!---http://www.cs.princeton.edu/courses/archive/fall11/cos323/notes/cos323_f11_lecture09_svd.pdf--->

<!---http://www.stats.uwo.ca/faculty/braun/ss3859/notes/Chapter4/ch4.pdf--->

<!---http://www.stat.berkeley.edu/~census/mlesan.pdf--->

<!---http://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node17.html---> <!--- this helped me get spline gradients--->

<!---https://scikit-sparse.readthedocs.io/en/latest/overview.html#developers--->

<!---https://vincentarelbundock.github.io/Rdatasets/datasets.html---> <!--- R Datasets!--->
