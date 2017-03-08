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

def gen_single_data(n=200):
    """
    1-dimensional problem
    """
    x = np.linspace(-5,5,n)[:,None]

    log_odds = -.5*x**2 + 5
    p = 1/(1+np.exp(-log_odds)).squeeze()
    obs = (np.random.rand(len(x)) < p).astype(np.int)

    lgam = LogisticGAM(lam=.6, n_iter=200, n_knots=20, spline_order=4)
    lgam.fit(x, obs)

    plt.figure()
    plt.plot(x, p, label='true probability', color='b', ls='--')
    plt.scatter(x, obs, label='observations', facecolor='None', color='k', marker='o', alpha='0.5')
    plt.plot(x, lgam.predict_proba(x), label='GAM probability', color='r')
    plt.legend(prop=fontP, bbox_to_anchor=(1.1, 1.05))
    plt.title('LogisticGAM on quadratic log-odds data')
    plt.savefig('imgs/pygam_single.png', dpi=300)


    plt.figure()
    plt.scatter(x, obs, facecolor='None', color='k', marker='o', alpha='0.5')
    plt.plot(x, lgam.predict_proba(x), color='r')
    plt.title('Accuracy: {}'.format(lgam.accuracy(X=x, y=obs)))
    plt.savefig('imgs/pygam_single_pred.png', dpi=300)

    ### UBRE Gridsearch
    scores = []
    lams = np.logspace(-2,2, 51)
    for lam in lams:
        lgam = LogisticGAM(lam=lam)
        lgam.fit(x, obs)
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
    multivariate problem
    """
    n = 5000
    x = np.random.rand(n,5) * 10 - 5
    cat = np.random.randint(0,4, n)
    x = np.c_[x, cat]
    log_odds = (-0.5*x[:,0]**2) + 5 +(-0.5*x[:,1]**2) + np.mod(x[:,-1], 2)*-30
    p = 1/(1+np.exp(-log_odds)).squeeze()

    obs = (np.random.rand(len(x)) < p).astype(np.int)

    lgam = LogisticGAM(lam=.6, n_iter=200, n_knots=20, spline_order=4)
    lgam.fit(x, obs)

    plt.figure()
    plt.plot(lgam.partial_dependence(np.sort(x, axis=0))[0])
    plt.savefig('imgs/pygam_multi_pdep.png', dpi=300)

if __name__ == '__main__':
    gen_single_data()
    gen_multi_data()
