import os

# Ensure NumPy doesn't secretly use all CPU cores
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from pygam import LinearGAM, PoissonGAM, s


class LinearGAMFit:
    """Benchmarks for Regression using synthetic data."""

    # Hard limits to keep the benchmark fast
    number = 1
    repeat = 3
    timeout = 60.0

    def setup(self):
        np.random.seed(42)
        # 3000 samples, 3 features (Matches Wage dataset size)
        self.X = np.random.rand(3000, 3)
        # Mathematical signal so the algorithm solves instantly
        self.y = self.X[:, 0] * 2 + self.X[:, 1] ** 2 + np.random.randn(3000) * 0.1

        self.gam = LinearGAM(s(0) + s(1) + s(2))
        self.gam_fitted = LinearGAM(s(0) + s(1) + s(2)).fit(self.X, self.y)
        self.lam_grid = np.logspace(-3, 3, 3)

    def time_fit(self):
        self.gam.fit(self.X, self.y)

    def time_predict(self):
        self.gam_fitted.predict(self.X)

    def time_gridsearch(self):
        self.gam.gridsearch(self.X, self.y, lam=self.lam_grid, progress=False)


class PoissonGAMFit:
    """Benchmarks for PoissonGAM using synthetic data."""

    number = 1
    repeat = 3
    timeout = 60.0

    def setup(self):
        np.random.seed(42)
        # 500 samples (Matches Coal dataset size)
        self.X = np.random.rand(500, 3)
        # Poisson-friendly signal
        expected_rate = np.exp(self.X[:, 0] * 0.5)
        self.y = np.random.poisson(lam=expected_rate)

        self.gam = PoissonGAM(s(0) + s(1) + s(2))

    def time_fit(self):
        self.gam.fit(self.X, self.y)
