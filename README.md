# pyGAM
Generalized Additive Models in Python.

<img src=imgs/pygam_single.png>

## About
Generalized Additize Models (GAMs) are smooth non-parametric models of the form:

![alt tag](http://latex.codecogs.com/svg.latex?g(\\mathbb{E}[y|X]) = \\beta_0 + f_1(X_1) + f_2(X_2) + \\dots + f_p(X_p))

where `X.T = [X_1, X_2, ..., X_p]` are independent variables, `y` is the dependent variable, and `g()` is the link function that relates our predictor variables to the expected value of the dependent variable.

The feature functions `f_i()` are built using **penalized regression splines**, which allow us to **automatically model non-linear relationships** without having to manually try out many different transformations on each variable.

GAMs extend standard linear models by allowing non-linear functions of features while maintaining additivity. Since the model is additive, it is easy to examine the effect of each `X_i` on `Y` individually while holding all other predictors constant.

The result is a very flexible model, where it is easy to incorporate prior knowledge and control overfitting.


## Regression
For **regression** problems, we can use a **linear GAM** which models:

![alt tag](http://latex.codecogs.com/svg.latex?\\mathbb{E}[y|X] = \\beta_0 + f_1(X_1) + f_2(X_2) + \\dots + f_p(X_p))

## Classification
For **binary classification** problems, we can use a **logistic GAM** which models:

![alt tag](http://latex.codecogs.com/svg.latex?log\\left
(\\frac{P(y=1|X)}{P(y=0|X)}\\right) = \\beta_0 + f_1(X_1) + f_2(X_2) + \\dots + f_p(X_p))

```python
from pygam import LogisticGAM

gam = LogisticGAM()
gam.fit(X, y)

plt.plot(X, gam.predict_proba(X), c='r')
plt.scatter(X, y, facecolor='None')
plt.title('Accuracy: {}'.format(gam.accuracy(X, y)))
```
<img src=imgs/pygam_single_pred.png>

## Penalties
With GAMs we can encode **prior knowledge** and **control overfitting** by using penalties. Common penalties include:

- second derivative smoothing
- harmonic smoothing
- monotonic smoothing

**Second derivative smoothing** is used by **default**, and ensures that the feature functions are not too wiggly.

## API
pyGAM is intuitive and adheres to a familiar API:

```python
from pygam import LogisticGAM

gam = LogisticGAM()
gam.fit(X_train, y_train)
```

Since GAMs are additive, it is also super easy to visualize each individual **feature function**, `f_i(X_i)`. These feature functions describe the effect of each `X_i` on `y` individually while marginalizing out all other predictors:

```python
pdeps = gam.partial_dependence(np.sort(X_train, axis=0))
plt.plot(pdeps)
```
<img src=imgs/pygam_multi_pdep.png>

## References
0. Hastie, Tibshirani, Friedman  
The Elements of Statistical Learning  
http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf  

0. James, Witten, Hastie and Tibshirani  
An Introduction to Statistical Learning  
http://www-bcf.usc.edu/~gareth/ISL/ISLR%20Sixth%20Printing.pdf  

0. Kim Larsen, 2015  
GAM: The Predictive Modeling Silver Bullet  
http://multithreaded.stitchfix.com/assets/files/gam.pdf  

0. Simon N. Wood, 2006  
Generalized Additive Models: an introduction with R  
<!---
http://reseau-mexico.fr/sites/reseau-mexico.fr/files/igam.pdf
--->

0. Deva Ramanan, 2008  
UCI Machine Learning: Notes on IRLS  
http://www.ics.uci.edu/~dramanan/teaching/ics273a_winter08/homework/irls_notes.pdf  

0. Paul Eilers & Brian Marx, 2015  
International Biometric Society: A Crash Course on P-splines  
http://www.ibschannel2015.nl/project/userfiles/Crash_course_handout.pdf


<!---http://www.cs.princeton.edu/courses/archive/fall11/cos323/notes/cos323_f11_lecture09_svd.pdf--->

<!---http://www.stats.uwo.ca/faculty/braun/ss3859/notes/Chapter4/ch4.pdf--->

<!---http://www.stat.berkeley.edu/~census/mlesan.pdf--->
