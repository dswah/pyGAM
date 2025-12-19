

# pyGAM

<a href="https://pygam.readthedocs.io/en/latest/?badge=latest"><img src=imgs/pygam_tensor.png width="250" align="right" /></a>

Generalized Additive Models in Python.

:rocket: **Version 0.12.0 out now!** [See release notes here](https://github.com/dswah/pyGAM/releases).

`pyGAM` is a package for building Generalized Additive Models in Python, with an emphasis on modularity and performance.

The API is designed for users of `scikit-learn` or `scipy`.


|  | **[Documentation](https://pygam.readthedocs.io/en/latest/?badge=latest)** · **[Tutorials](https://pygam.readthedocs.io/en/latest/notebooks/tour_of_pygam.html)** · **[Medium article](https://medium.com/just-another-data-scientist/building-interpretable-models-with-generalized-additive-models-in-python-c4404eaf5515)** |
|---|---|
| **Open&#160;Source** | [![Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/dswah/pygam/blob/main/LICENSE) [![GC.OS Sponsored](https://img.shields.io/badge/GC.OS-Sponsored%20Project-orange.svg?style=flat&colorA=0eac92&colorB=2077b4)](https://gc-os-ai.github.io/) |
| **Community** | [![!discord](https://img.shields.io/static/v1?logo=discord&label=discord&message=chat&color=lightgreen)](https://discord.gg/Rt8By5Jj) [![!slack](https://img.shields.io/static/v1?logo=linkedin&label=LinkedIn&message=news&color=lightblue)](https://www.linkedin.com/company/german-center-for-open-source-ai) |
| **CI/CD** | [![github-actions](https://img.shields.io/github/actions/workflow/status/dswah/pygam/pypi.yml?logo=github)](https://github.com/dswah/pygam/actions/workflows/pypi.yml) [![readthedocs](https://img.shields.io/readthedocs/pygam?logo=readthedocs)](https://pygam.readthedocs.io/en/latest/?badge=latest) |
| **Code** |  [![!pypi](https://img.shields.io/pypi/v/pygam?color=orange)](https://pypi.org/project/pygam/) [![!conda](https://img.shields.io/conda/vn/conda-forge/pygam)](https://anaconda.org/conda-forge/pygam) [![!python-versions](https://img.shields.io/pypi/pyversions/pygam)](https://www.python.org/) [![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  |
| **Downloads** | ![PyPI - Downloads](https://img.shields.io/pypi/dw/pygam) ![PyPI - Downloads](https://img.shields.io/pypi/dm/pygam) [![Downloads](https://static.pepy.tech/personalized-badge/pygam?period=total&units=international_system&left_color=grey&right_color=blue&left_text=cumulative%20(pypi))](https://pepy.tech/project/pygam) |
| **Citation** | [![!zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.1208723.svg)](https://doi.org/10.5281/zenodo.1208723) |

## Documentation
- [Official pyGAM Documentation: Read the Docs](https://pygam.readthedocs.io/en/latest/?badge=latest)
- [Building interpretable models with Generalized additive models in Python](https://medium.com/just-another-data-scientist/building-interpretable-models-with-generalized-additive-models-in-python-c4404eaf5515)

## Installation
```pip install pygam```

### Acceleration
Most of pyGAM's computations are linear algebra operations.

To speed up optimization on large models with constraints, it helps to have [intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) installed.

It is currently a bit tricky to install a Numpy linked to the MKL routines with Conda because you have to be careful with which channel you are using. Pip's Numpy-MKL is outdated.

An alternative is to use a [third-party build](https://urob.github.io/numpy-mkl):
```
pip install numpy scipy --extra-index-url https://urob.github.io/numpy-mkl
```

## Contributing - HELP REQUESTED
Contributions are most welcome!

You can help pyGAM in many ways including:

- Working on a [known bug](https://github.com/dswah/pyGAM/labels/bug).
- Trying it out and reporting bugs or what was difficult.
- Helping improve the documentation.
- Writing new [distributions](https://github.com/dswah/pyGAM/blob/main/pygam/distributions.py), and [link functions](https://github.com/dswah/pyGAM/blob/main/pygam/links.py).
- If you need some ideas, please take a look at the [issues](https://github.com/dswah/pyGAM/issues).


To start:
- **fork the project** and cut a new branch
- **install** `pygam`, editable with developer **dependencies** (in a new python environment)

```
pip install --upgrade pip
pip install -e ".[dev]"
```

Make some changes and write a test...
- **Test** your contribution (eg from the `.../pyGAM`):
```py.test -s```
- When you are happy with your changes, make a **pull request** into the `main` branch of the main project.


## About
Generalized Additive Models (GAMs) are smooth semi-parametric models of the form:

$$g\left(\mathbb{E}[y|X]\right)=\beta_0+f_1(X_1)+f_2(X_2)+\dots+f_p(X_p)$$


where $X = [X_1, X_2, ..., X_p]$ are independent variables, $y$ is the dependent variable, and $g$ is a link function that relates our predictor variables to the expected value of the dependent variable.


The feature functions $f_i$ are built using **penalized B-splines**, which allow us to **automatically model non-linear relationships** without having to manually try out many different transformations on each variable.

<img src=imgs/pygam_basis.png>

GAMs extend generalized linear models by allowing non-linear functions of features while maintaining additivity.

Since GAMs are additive, it is easy to examine the effect of each $X_i$ on $y$ individually while holding all other predictors constant.

As a result, GAMs are a class of very flexible and interpretable models, which also make it is easy to incorporate prior knowledge and control overfitting.

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
https://www.sas.upenn.edu/~fdiebold/NoHesitations/BookAdvanced.pdf

0. James, Witten, Hastie, Tibshirani, and Taylor
An Introduction to Statistical Learning with Applications in Python
https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html

0. Paul Eilers & Brian Marx, 1996
Flexible Smoothing with B-splines and Penalties
https://sites.stat.washington.edu/courses/stat527/s14/readings/EilersMarx_StatSci_1996.pdf

0. Kim Larsen, 2015
GAM: The Predictive Modeling Silver Bullet
http://multithreaded.stitchfix.com/assets/files/gam.pdf

0. Paul Eilers, Brian Marx, and Maria Durbán, 2015
Twenty years of P-splines
https://e-archivo.uc3m.es/rest/api/core/bitstreams/4e23bd9f-c90d-4598-893e-deb0a6bf0728/content

0. Keiding, Niels, 1991
Age-specific incidence and prevalence: a statistical perspective
https://academic.oup.com/jrsssa/article-abstract/154/3/371/7106499


<!---http://www.cs.princeton.edu/courses/archive/fall11/cos323/notes/cos323_f11_lecture09_svd.pdf--->

<!---http://www.stats.uwo.ca/faculty/braun/ss3859/notes/Chapter4/ch4.pdf--->

<!---http://www.stat.berkeley.edu/~census/mlesan.pdf--->

<!---http://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node17.html---> <!--- this helped me get spline gradients--->

<!---https://scikit-sparse.readthedocs.io/en/latest/overview.html#developers--->

<!---https://vincentarelbundock.github.io/Rdatasets/datasets.html---> <!--- R Datasets!--->
