import numpy as np
from pygam import LinearGAM

X = np.random.rand(100, 1)
y = np.sin(X[:, 0])

gam = LinearGAM().fit(X, y)
params = gam.get_params()

print("coef_ in params:", "coef_" in params)