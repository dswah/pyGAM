# pyGAM
Generalized Additive Models in Python

## About
Models of the form
![alt tag](http://latex.codecogs.com/svg.latex?g(\mathbb{E}[y]) = \alpha + s_1(x_1) + \dots + s_p(x_p))
where `y` is the dependent variable, and `g()` is the link function that links our predictor variables to the expected value of the dependent variable.

For a binary classification problem, we use the logit link function:
![alt tag](http://latex.codecogs.com/svg.latex?g(\mathbb{E}[y]) = log(\\frac{P(y=1)}{P(y=0)}))

## References
0. Hastie, Tibshirani,Friedman
The Elements of Statistical Learning
http://www-stat.stanford.edu/~tibs/ElemStatLearn/download.html

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
