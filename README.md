# pyGAM
Generalized Additive Models in Python

## About
GAMs are smooth non-parametric models of the form:

![alt tag](http://latex.codecogs.com/svg.latex?g(\\mathbb{E}[y|X]) = \\beta_0 + f_1(X_1) + f_2(X_2) + \\dots + f_p(X_p))

where `X.T = [X_1, X_2, ..., X_p]` are our independent variables, `y` is the dependent variable, and `g()` is the link function that links our predictor variables to the expected value of the dependent variable.

For a **regression** problem, we can use a linear GAM which models:

![alt tag](http://latex.codecogs.com/svg.latex?\\mathbb{E}[y|X] = \\beta_0 + f_1(X_1) + f_2(X_2) + \\dots + f_p(X_p))


For a **binary classification** problem, we can use a logistic GAM which models:

![alt tag](http://latex.codecogs.com/svg.latex?log\\left
(\\frac{P(y=1|X)}{P(y=0|X)}\\right) = \\beta_0 + f_1(X_1) + f_2(X_2) + \\dots + f_p(X_p))

## References
0. Hastie, Tibshirani,Friedman  
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
