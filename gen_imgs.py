"""
generate some plots for the pyGAM repo
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from pygam import *

np.random.seed(420)
fontP = FontProperties()
fontP.set_size('small')

def gen_basis_fns():
  x = np.linspace(0,7,100)[:,None]
  x2 = np.linspace(0,7,500)[:,None]
  y = np.sin(x*2*np.pi)*x + 5*x + 1.5*np.random.randn(len(x),1)# rising sin

  gam = LinearGAM(fit_intercept=False, fit_linear=False)
  gam.gridsearch(x, y)

  plt.figure()
  fig, ax = plt.subplots(2,1)
  ax[0].plot(x2, gam._modelmat(x2, feature=0).todense().A);
  ax[0].set_title('b-Spline Basis Functions')

  ax[1].scatter(x, y, facecolor='None')
  ax[1].plot(x2, (gam._modelmat(x2, feature=0).todense().A * gam._b));
  ax[1].plot(x2, gam.predict(x2), 'k')
  ax[1].set_title('Fitted Model')
  ax[1].set_xlim([0,7])
  plt.savefig('imgs/pygam_basis.png', dpi=300)

def gen_single_data_linear(n=500):
  x = np.linspace(0,7,n)[:,None]
  y = np.sin(x*2*np.pi)*x + 5*x + 1.5*np.random.randn(len(x),1)# rising sin

  gam = LinearGAM()
  gam.gridsearch(x, y)

  # single pred linear
  plt.figure()
  plt.scatter(x, y, facecolor='None')
  plt.plot(x, gam.predict(x), color='r')
  plt.title('Best Lambda: {}'.format(gam.lam))
  plt.savefig('imgs/pygam_single_pred_linear.png', dpi=300)

def gen_single_data(n=200):
    """
    1-dimensional Logistic problem
    """
    x = np.linspace(-5,5,n)[:,None]

    log_odds = -.5*x**2 + 5
    p = 1/(1+np.exp(-log_odds)).squeeze()
    y = (np.random.rand(len(x)) < p).astype(np.int)

    lgam = LogisticGAM()
    lgam.fit(x, y)

    # title plot
    plt.figure()
    plt.plot(x, p, label='true probability', color='b', ls='--')
    plt.scatter(x, y, label='observations', facecolor='None')
    plt.plot(x, lgam.predict_proba(x), label='GAM probability', color='r')
    plt.legend(prop=fontP, bbox_to_anchor=(1.1, 1.05))
    plt.title('LogisticGAM on quadratic log-odds data')
    plt.savefig('imgs/pygam_single.png', dpi=300)

    # single pred
    plt.figure()
    plt.scatter(x, y, facecolor='None')
    plt.plot(x, lgam.predict_proba(x), color='r')
    plt.title('Accuracy: {}'.format(lgam.accuracy(X=x, y=y)))
    plt.savefig('imgs/pygam_single_pred.png', dpi=300)

    # UBRE Gridsearch
    scores = []
    lams = np.logspace(-4,2, 51)
    for lam in lams:
        lgam = LogisticGAM(lam=lam)
        lgam.fit(x, y)
        scores.append(lgam._statistics['UBRE'])
    best = np.argmin(scores)

    plt.figure()
    plt.plot(lams, scores)
    plt.scatter(lams[best], scores[best], facecolor='None')
    plt.xlabel('$\lambda$')
    plt.ylabel('UBRE')
    plt.title('Best $\lambda$: %.3f'% lams[best])

    plt.savefig('imgs/pygam_lambda_gridsearch.png', dpi=300)


def gen_multi_data(n=200):
    """
    multivariate Logistic problem
    """
    n = 5000
    x = np.random.rand(n,5) * 10 - 5
    cat = np.random.randint(0,4, n)
    x = np.c_[x, cat]
    log_odds = (-0.5*x[:,0]**2) + 5 +(-0.5*x[:,1]**2) + np.mod(x[:,-1], 2)*-30
    p = 1/(1+np.exp(-log_odds)).squeeze()

    obs = (np.random.rand(len(x)) < p).astype(np.int)

    lgam = LogisticGAM()
    lgam.fit(x, obs)

    plt.figure()
    plt.plot(lgam.partial_dependence(np.sort(x, axis=0)))
    plt.savefig('imgs/pygam_multi_pdep.png', dpi=300)

if __name__ == '__main__':
    gen_single_data()
    gen_multi_data()
    gen_single_data_linear()
    gen_basis_fns()
