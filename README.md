[![Build Status](https://travis-ci.org/dswah/pyGAM.svg?branch=master)](https://travis-ci.org/dswah/pyGAM)
[![PyPI version](https://badge.fury.io/py/pygam.svg)](https://badge.fury.io/py/pygam)
[![codecov](https://codecov.io/gh/dswah/pygam/branch/master/graph/badge.svg)](https://codecov.io/gh/dswah/pygam)
[![python27](https://img.shields.io/badge/python-2.7-blue.svg)](https://badge.fury.io/py/pygam)
[![python36](https://img.shields.io/badge/python-3.6-blue.svg)](https://badge.fury.io/py/pygam)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1208723.svg)](https://doi.org/10.5281/zenodo.1208723)


# pyGAM
Generalized Additive Models in Python.

<!--<img src=imgs/pygam_single.png>-->
<img src=imgs/pygam_cake_data.png>

## Tutorial
[pyGAM: Getting started with Generalized Additive Models in Python](https://medium.com/@jpoberhauser/pygam-getting-started-with-generalized-additive-models-in-python-457df5b4705f)

## Installation
```pip install pygam```

### scikit-sparse
To speed up optimization on large models with constraints, it helps to have `scikit-sparse` installed because it contains a slightly faster, sparse version of Cholesky factorization. The import from `scikit-sparse` references `nose`, so you'll need that too.

The easiest way is to use Conda:  
```conda install -c conda-forge scikit-sparse```

[scikit-sparse docs](http://pythonhosted.org/scikit-sparse/overview.html#download)

## Contributing
Contributions are most welcome!

You can help pyGAM in many ways including:

- Trying it out and reporting bugs or what was difficult.
- Helping improve the documentation.
- Writing new [distributions](https://github.com/dswah/pyGAM/blob/master/pygam/distributions.py), and [link functions](https://github.com/dswah/pyGAM/blob/master/pygam/links.py).
- If you need some ideas, please take a look at the [issues](https://github.com/dswah/pyGAM/issues).


To start:
- **fork the project** and cut a new branch
- Now **install** the testing **dependencies**

```
conda install pytest numpy pandas scipy pytest-cov cython scikit-sparse
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


## Regression
For **regression** problems, we can use a **linear GAM** which models:

![alt tag](http://latex.codecogs.com/svg.latex?\mathbb{E}[y|X]=\beta_0+f_1(X_1)+f_2(X_2)+\dots+f_p(X_p))

```python
from pygam import LinearGAM
from pygam.datasets import wage

X, y = wage(return_X_y=True)

gam = LinearGAM(n_splines=10).gridsearch(X, y)
XX = gam.generate_X_grid()

fig, axs = plt.subplots(1, 3)
titles = ['year', 'age', 'education']

for i, ax in enumerate(axs):
    pdep, confi = gam.partial_dependence(XX, feature=i, width=.95)

    ax.plot(XX[:, i], pdep)
    ax.plot(XX[:, i], *confi, c='r', ls='--')
    ax.set_title(titles[i])
```
<img src=imgs/pygam_wage_data_linear.png>

Even though we allowed **n_splines=10** per numerical feature, our **smoothing penalty** reduces us to just 14 **effective degrees of freedom**:

```
gam.summary()

LinearGAM                                                                                                 
=============================================== ==========================================================
Distribution:                        NormalDist Effective DoF:                                      13.532
Link Function:                     IdentityLink Log Likelihood:                                -24119.2334
Number of Samples:                         3000 AIC:                                            48267.5307
                                                AICc:                                            48267.682
                                                GCV:                                             1247.0706
                                                Scale:                                           1236.9495
                                                Pseudo R-Squared:                                   0.2926
==========================================================================================================
Feature Function   Data Type      Num Splines   Spline Order  Linear Fit  Lambda     P > x      Sig. Code
================== ============== ============= ============= =========== ========== ========== ==========
feature 1          numerical      10            3             False       15.8489    1.63e-03   **        
feature 2          numerical      10            3             False       15.8489    1.50e-11   ***       
feature 3          categorical    5             0             False       15.8489    1.25e-14   ***       
intercept                                                                            1.11e-16   ***       
==========================================================================================================
Significance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```


With **LinearGAMs**, we can also check the **prediction intervals**:

```python
from pygam import LinearGAM
from pygam.datasets import mcycle

X, y = mcycle(return_X_y=True)

gam = LinearGAM().gridsearch(X, y)
XX = gam.generate_X_grid()

plt.plot(XX, gam.predict(XX), 'r--')
plt.plot(XX, gam.prediction_intervals(XX, width=.95), color='b', ls='--')

plt.scatter(X, y, facecolor='gray', edgecolors='none')
plt.title('95% prediction interval')
```
<img src=imgs/pygam_mcycle_data_linear.png>

And simulate from the posterior:

```python
# continuing last example with the mcycle dataset
for response in gam.sample(X, y, quantity='y', n_draws=50, sample_at_X=XX):
    plt.scatter(XX, response, alpha=.03, color='k')
plt.plot(XX, gam.predict(XX), 'r--')
plt.plot(XX, gam.prediction_intervals(XX, width=.95), color='b', ls='--')
plt.title('draw samples from the posterior of the coefficients')
```

<img src=imgs/pygam_mcycle_data_linear_sample_from_posterior.png>

## Classification
For **binary classification** problems, we can use a **logistic GAM** which models:

![alt tag](http://latex.codecogs.com/svg.latex?log\left(\frac{P(y=1|X)}{P(y=0|X)}\right)=\beta_0+f_1(X_1)+f_2(X_2)+\dots+f_p(X_p))

```python
from pygam import LogisticGAM
from pygam.datasets import default

X, y = default(return_X_y=True)

gam = LogisticGAM().gridsearch(X, y)
XX = gam.generate_X_grid()

fig, axs = plt.subplots(1, 3)
titles = ['student', 'balance', 'income']

for i, ax in enumerate(axs):
    pdep, confi = gam.partial_dependence(XX, feature=i, width=.95)

    ax.plot(XX[:, i], pdep)
    ax.plot(XX[:, i], confi[0], c='r', ls='--')
    ax.set_title(titles[i])    
```
<img src=imgs/pygam_default_data_logistic.png>

We can then check the accuracy:

```python
gam.accuracy(X, y)

0.97389999999999999
```

Since the **scale** of the **Binomial distribution** is known, our gridsearch minimizes the **Un-Biased Risk Estimator** (UBRE) objective:

```
gam.summary()

LogisticGAM                                                                                               
=============================================== ==========================================================
Distribution:                      BinomialDist Effective DoF:                                      4.3643
Link Function:                        LogitLink Log Likelihood:                                  -788.7121
Number of Samples:                        10000 AIC:                                             1586.1527
                                                AICc:                                            1586.1595
                                                UBRE:                                                2.159
                                                Scale:                                                 1.0
                                                Pseudo R-Squared:                                   0.4599
==========================================================================================================
Feature Function   Data Type      Num Splines   Spline Order  Linear Fit  Lambda     P > x      Sig. Code
================== ============== ============= ============= =========== ========== ========== ==========
feature 1          categorical    2             0             False       1000.0     4.41e-03   **        
feature 2          numerical      25            3             False       1000.0     0.00e+00   ***       
feature 3          numerical      25            3             False       1000.0     2.35e-02   *         
intercept                                                                            0.00e+00   ***       
==========================================================================================================
Significance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```


## Poisson and Histogram Smoothing
We can intuitively perform **histogram smoothing** by modeling the counts in each bin
as being distributed Poisson via **PoissonGAM**.

```python
from pygam import PoissonGAM
from pygam.datasets import faithful

X, y = faithful(return_X_y=True)

gam = PoissonGAM().gridsearch(X, y)

plt.hist(faithful(return_X_y=False)['eruptions'], bins=200, color='k');
plt.plot(X, gam.predict(X), color='r')
plt.title('Best Lambda: {0:.2f}'.format(gam.lam))
```
<img src=imgs/pygam_poisson.png>


## Custom Models
It's also easy to build custom models, by using the base **GAM** class and specifying the **distribution** and the **link function**.

```python
from pygam import GAM
from pygam.datasets import trees

X, y = trees(return_X_y=True)

gam = GAM(distribution='gamma', link='log', n_splines=4)
gam.gridsearch(X, y)

plt.scatter(y, gam.predict(X))
plt.xlabel('true volume')
plt.ylabel('predicted volume')
```
<img src=imgs/pygam_custom.png>

We can check the quality of the fit by looking at the `Pseudo R-Squared`:

```
gam.summary()

GAM                                                                                                       
=============================================== ==========================================================
Distribution:                         GammaDist Effective DoF:                                      4.1544
Link Function:                          LogLink Log Likelihood:                                   -66.9372
Number of Samples:                           31 AIC:                                              144.1834
                                                AICc:                                             146.7369
                                                GCV:                                                0.0095
                                                Scale:                                              0.0073
                                                Pseudo R-Squared:                                   0.9767
==========================================================================================================
Feature Function   Data Type      Num Splines   Spline Order  Linear Fit  Lambda     P > x      Sig. Code
================== ============== ============= ============= =========== ========== ========== ==========
feature 1          numerical      4             3             False       0.0158     3.42e-12   ***       
feature 2          numerical      4             3             False       0.0158     1.29e-09   ***       
intercept                                                                            7.60e-13   ***       
==========================================================================================================
Significance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

## Penalties / Constraints
With GAMs we can encode **prior knowledge** and **control overfitting** by using penalties and constraints.

#### Available penalties:
- second derivative smoothing (default on numerical features)
- L2 smoothing (default on categorical features)

#### Availabe constraints:
- monotonic increasing/decreasing smoothing
- convex/concave smoothing
- periodic smoothing [soon...]


We can inject our intuition into our model by using **monotonic** and **concave** constraints:

```python
from pygam import LinearGAM
from pygam.datasets import hepatitis

X, y = hepatitis(return_X_y=True)

gam1 = LinearGAM(constraints='monotonic_inc').fit(X, y)
gam2 = LinearGAM(constraints='concave').fit(X, y)

fig, ax = plt.subplots(1, 2)
ax[0].plot(X, y, label='data')
ax[0].plot(X, gam1.predict(X), label='monotonic fit')
ax[0].legend()

ax[1].plot(X, y, label='data')
ax[1].plot(X, gam2.predict(X), label='concave fit')
ax[1].legend()
```
<img src=imgs/pygam_constraints.png>

## API
pyGAM is intuitive, modular, and adheres to a familiar API:

```python
from pygam import LogisticGAM
from pygam.datasets import toy_classification

X, y = toy_classification(return_X_y=True)

gam = LogisticGAM()
gam.fit(X, y)
```

Since GAMs are additive, it is also super easy to visualize each individual **feature function**, `f_i(X_i)`. These feature functions describe the effect of each `X_i` on `y` individually while marginalizing out all other predictors:

```python
pdeps = gam.partial_dependence(X)
plt.plot(pdeps)
```
<img src=imgs/pygam_multi_pdep.png>

## Current Features
### Models
pyGAM comes with many models out-of-the-box:

- GAM (base class for constructing custom models)
- LinearGAM
- LogisticGAM
- GammaGAM
- PoissonGAM
- InvGaussGAM

You can mix and match distributions with link functions to create custom models!

```python
gam = GAM(distribution='gamma', link='inverse')
```

### Distributions

- Normal
- Binomial
- Gamma
- Poisson
- Inverse Gaussian

### Link Functions
Link functions take the distribution mean to the linear prediction. These are the canonical link functions for the above distributions:

- Identity
- Logit
- Inverse
- Log
- Inverse-squared

### Callbacks
Callbacks are performed during each optimization iteration. It's also easy to write your own.

- deviance - model deviance
- diffs - differences of coefficient norm
- accuracy - model accuracy for LogisticGAM
- coef - coefficient logging

You can check a callback by inspecting:

```python
plt.plot(gam.logs_['deviance'])
```
<img src=imgs/pygam_multi_deviance.png>

### Linear Extrapolation
<img src=imgs/pygam_mcycle_data_extrapolation.png>

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
