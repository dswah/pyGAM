import numpy as np
from pygam.links import LogitLink
from pygam.distributions import BinomialDist

link = LogitLink()
dist = BinomialDist()

lp = np.array([-1000, -10, 0, 10, 1000])

print(link.mu(lp, dist))