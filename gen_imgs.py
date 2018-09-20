"""
generate some plots for the pyGAM repo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits import mplot3d

from pygam import *
from pygam.datasets import hepatitis, wage, faithful, mcycle, trees, default, cake, toy_classification, toy_interaction, chicago

np.random.seed(420)
fontP = FontProperties()
fontP.set_size('small')

# poisson, histogram smoothing
# custom GAM tree dataset
# new basis function thing
# wage dataset to illustrate partial dependence
# monotonic increasing, concave constraint on hep data
# prediction intervals on motorcycle data

def gen_basis_fns():
    X, y = hepatitis()
    gam = LinearGAM(lam=.6, fit_intercept=False).fit(X, y)
    XX = gam.generate_X_grid(term=0, n=500)

    plt.figure()
    fig, ax = plt.subplots(2,1)
    ax[0].plot(XX, gam._modelmat(XX, term=0).A);
    ax[0].set_title('b-Spline Basis Functions')

    ax[1].scatter(X, y, facecolor='gray', edgecolors='none')
    ax[1].plot(XX, gam._modelmat(XX).A * gam.coef_);
    ax[1].plot(XX, gam.predict(XX), 'k')
    ax[1].set_title('Fitted Model')
    fig.tight_layout()
    plt.savefig('imgs/pygam_basis.png', dpi=300)

def cake_data_in_one():
    X, y = cake()

    gam = LinearGAM(fit_intercept=True)
    gam.gridsearch(X,y)

    XX = gam.generate_X_grid()

    plt.figure()
    plt.plot(gam.partial_dependence(XX))
    plt.title('LinearGAM')
    plt.savefig('imgs/pygam_cake_data.png', dpi=300)

def faithful_data_poisson():
    X, y = faithful()
    gam = PoissonGAM().gridsearch(X, y)

    plt.figure()
    plt.hist(faithful(return_X_y=False)['eruptions'], bins=200, color='k');

    plt.plot(X, gam.predict(X), color='r')
    plt.title('Best Lambda: {0:.2f}'.format(gam.lam[0][0]))
    plt.savefig('imgs/pygam_poisson.png', dpi=300)

def single_data_linear():
    X, y = mcycle()

    gam = LinearGAM()
    gam.gridsearch(X, y)

    # single pred linear
    plt.figure()
    plt.scatter(X, y, facecolor='gray', edgecolors='none')
    plt.plot(X, gam.predict(X), color='r')
    plt.title('Best Lambda: {0:.2f}'.format(gam.lam))
    plt.savefig('imgs/pygam_single_pred_linear.png', dpi=300)

def mcycle_data_linear():
    X, y = mcycle()

    gam = LinearGAM()
    gam.gridsearch(X, y)

    XX = gam.generate_X_grid(term=0)
    plt.figure()
    plt.scatter(X, y, facecolor='gray', edgecolors='none')
    plt.plot(XX, gam.predict(XX), 'r--')
    plt.plot(XX, gam.prediction_intervals(XX, width=.95), color='b', ls='--')
    plt.title('95% prediction interval')

    plt.savefig('imgs/pygam_mcycle_data_linear.png', dpi=300)


    m = X.min()
    M = X.max()
    XX = np.linspace(m - 10, M + 10, 500)
    Xl = np.linspace(m - 10, m, 50)
    Xr = np.linspace(M, M + 10, 50)

    plt.figure()

    plt.plot(XX, gam.predict(XX), 'k')
    plt.plot(Xl, gam.confidence_intervals(Xl), color='b', ls='--')
    plt.plot(Xr, gam.confidence_intervals(Xr), color='b', ls='--')
    plt.plot(X, gam.confidence_intervals(X), color='r', ls='--')

    plt.savefig('imgs/pygam_mcycle_data_extrapolation.png', dpi=300)

def wage_data_linear():
    X, y = wage()

    gam = LinearGAM(s(0) + s(1) + f(2))
    gam.gridsearch(X, y, lam=np.logspace(-5,3,50))

    plt.figure()
    fig, axs = plt.subplots(1,3)

    titles = ['year', 'age', 'education']
    for i, ax in enumerate(axs):
        XX = gam.generate_X_grid(term=i)
        ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
        ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1],
                c='r', ls='--')
        if i == 0:
            ax.set_ylim(-30,30);
        ax.set_title(titles[i])

    fig.tight_layout()
    plt.savefig('imgs/pygam_wage_data_linear.png', dpi=300)

def default_data_logistic():
    X, y = default()

    gam = LogisticGAM(f(0) + s(1) + s(2))
    gam.gridsearch(X, y)

    plt.figure()
    fig, axs = plt.subplots(1,3)

    titles = ['student', 'balance', 'income']
    for i, ax in enumerate(axs):
        XX = gam.generate_X_grid(term=i)

        ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
        ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1],
                c='r', ls='--')
        ax.set_title(titles[i])

    fig.tight_layout()
    plt.savefig('imgs/pygam_default_data_logistic.png', dpi=300)

def constraints():
    X, y = hepatitis(return_X_y=True)

    gam1 = LinearGAM(s(0, constraints='monotonic_inc')).fit(X, y)
    gam2 = LinearGAM(s(0, constraints='concave')).fit(X, y)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(X, y, label='data')
    ax[0].plot(X, gam1.predict(X), label='monotonic fit')
    ax[0].legend()

    ax[1].plot(X, y, label='data')
    ax[1].plot(X, gam2.predict(X), label='concave fit')
    ax[1].legend()

    fig.tight_layout()
    plt.savefig('imgs/pygam_constraints.png', dpi=300)

def trees_data_custom():
    X, y = trees()
    gam = GAM(distribution='gamma', link='log')
    gam.gridsearch(X, y)

    plt.figure()
    plt.scatter(y, gam.predict(X))
    plt.xlabel('true volume')
    plt.ylabel('predicted volume')
    plt.savefig('imgs/pygam_custom.png', dpi=300)

# def gen_single_data(n=200):
#     """
#     1-dimensional Logistic problem
#     """
#     x = np.linspace(-5,5,n)[:,None]
#
#     log_odds = -.5*x**2 + 5
#     p = 1/(1+np.exp(-log_odds)).squeeze()
#     y = (np.random.rand(len(x)) < p).astype(np.int)
#
#     lgam = LogisticGAM()
#     lgam.fit(x, y)
#
#     # title plot
#     plt.figure()
#     plt.plot(x, p, label='true probability', color='b', ls='--')
#     plt.scatter(x, y, label='observations', facecolor='None')
#     plt.plot(x, lgam.predict_proba(x), label='GAM probability', color='r')
#     plt.legend(prop=fontP, bbox_to_anchor=(1.1, 1.05))
#     plt.title('LogisticGAM on quadratic log-odds data')
#     plt.savefig('imgs/pygam_single.png', dpi=300)
#
#     # single pred
#     plt.figure()
#     plt.scatter(x, y, facecolor='None')
#     plt.plot(x, lgam.predict_proba(x), color='r')
#     plt.title('Accuracy: {}'.format(lgam.accuracy(X=x, y=y)))
#     plt.savefig('imgs/pygam_single_pred.png', dpi=300)
#
#     # UBRE Gridsearch
#     scores = []
#     lams = np.logspace(-4,2, 51)
#     for lam in lams:
#         lgam = LogisticGAM(lam=lam)
#         lgam.fit(x, y)
#         scores.append(lgam.statistics_['UBRE'])
#     best = np.argmin(scores)
#
#     plt.figure()
#     plt.plot(lams, scores)
#     plt.scatter(lams[best], scores[best], facecolor='None')
#     plt.xlabel('$\lambda$')
#     plt.ylabel('UBRE')
#     plt.title('Best $\lambda$: %.3f'% lams[best])
#
#     plt.savefig('imgs/pygam_lambda_gridsearch.png', dpi=300)


def gen_multi_data(n=5000):
    """
    multivariate Logistic problem
    """
    X, y = toy_classification(return_X_y=True, n=10000)

    lgam = LogisticGAM(s(0) + s(1) + s(2) + s(3) + s(4) + f(5))
    lgam.fit(X, y)

    plt.figure()
    for i, term in enumerate(lgam.terms):
        if term.isintercept:
            continue
        plt.plot(lgam.partial_dependence(term=i))

    plt.savefig('imgs/pygam_multi_pdep.png', dpi=300)

    plt.figure()
    plt.plot(lgam.logs_['deviance'])
    plt.savefig('imgs/pygam_multi_deviance.png', dpi=300)

def gen_tensor_data():
    """
    toy interaction data
    """
    X, y = toy_interaction(return_X_y=True, n=10000)

    gam = LinearGAM(te(0, 1,lam=0.1)).fit(X, y)

    XX = gam.generate_X_grid(term=0, meshgrid=True)
    Z = gam.partial_dependence(term=0, meshgrid=True)

    fig = plt.figure(figsize=(9,6))
    ax = plt.axes(projection='3d')
    ax.dist = 7.5
    ax.plot_surface(XX[0], XX[1], Z, cmap='viridis')
    ax.set_axis_off()
    fig.tight_layout()
    plt.savefig('imgs/pygam_tensor.png', transparent=True, dpi=300)

def chicago_tensor():
    """
    chicago tensor
    """
    X, y = chicago()
    gam = PoissonGAM(s(0, n_splines=200) + te(3, 1) + s(2)).fit(X, y)

    XX = gam.generate_X_grid(term=1, meshgrid=True)
    Z = gam.partial_dependence(term=1, meshgrid=True)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(XX[0], XX[1], Z, cmap='viridis')
    fig.tight_layout()

    plt.savefig('imgs/pygam_chicago_tensor.png', dpi=300)


def expectiles():
    """
    a bunch of expectiles
    """

    X, y = mcycle(return_X_y=True)

    # lets fit the mean model first by CV
    gam50 = ExpectileGAM(expectile=0.5).gridsearch(X, y)

    # and copy the smoothing to the other models
    lam = gam50.lam

    # now fit a few more models
    gam95 = ExpectileGAM(expectile=0.95, lam=lam).fit(X, y)
    gam75 = ExpectileGAM(expectile=0.75, lam=lam).fit(X, y)
    gam25 = ExpectileGAM(expectile=0.25, lam=lam).fit(X, y)
    gam05 = ExpectileGAM(expectile=0.05, lam=lam).fit(X, y)

    XX = gam50.generate_X_grid(term=0, n=500)

    fig = plt.figure()
    plt.scatter(X, y, c='k', alpha=0.2)
    plt.plot(XX, gam95.predict(XX), label='0.95')
    plt.plot(XX, gam75.predict(XX), label='0.75')
    plt.plot(XX, gam50.predict(XX), label='0.50')
    plt.plot(XX, gam25.predict(XX), label='0.25')
    plt.plot(XX, gam05.predict(XX), label='0.05')
    plt.legend()
    fig.tight_layout()

    plt.savefig('imgs/pygam_expectiles.png', dpi=300)


if __name__ == '__main__':
    gen_basis_fns()
    faithful_data_poisson()
    wage_data_linear()
    default_data_logistic()
    constraints()
    trees_data_custom()
    mcycle_data_linear()
    # cake_data_in_one()
    gen_multi_data()
    gen_tensor_data()
    chicago_tensor()
    expectiles()
