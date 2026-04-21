import os

# Lock threads to 1 for deterministic benchmarking across environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from pygam import LinearGAM, PoissonGAM, s


class LinearGAMFit:
    """Macro-benchmarks for LinearGAM training and inference."""

    number = 1
    repeat = 3
    timeout = 60.0

    def setup(self):
        # Reproducing Synthetic data
        np.random.seed(42)
        self.X = np.random.rand(3000, 3)
        self.y = self.X[:, 0] * 2 + self.X[:, 1] ** 2 + np.random.randn(3000) * 0.1

        self.gam = LinearGAM(s(0) + s(1) + s(2))
        self.gam_fitted = LinearGAM(s(0) + s(1) + s(2)).fit(self.X, self.y)
        self.lam_grid = np.logspace(-3, 3, 3)

    def time_fit(self):
        # Measures the time of the core fitting logic
        self.gam.fit(self.X, self.y)

    def time_predict(self):
        # Measures inference speed
        self.gam_fitted.predict(self.X)

    def time_gridsearch(self):
        # Measures hyperparameter tuning overhead
        self.gam.gridsearch(self.X, self.y, lam=self.lam_grid, progress=False)


class PoissonGAMFit:
    """Macro-benchmarks for PoissonGAM (tests the iterative PIRLS loop)."""

    number = 1
    repeat = 3
    timeout = 60.0

    def setup(self):
        np.random.seed(42)
        self.X = np.random.rand(500, 3)
        expected_rate = np.exp(self.X[:, 0] * 0.5)
        self.y = np.random.poisson(lam=expected_rate)
        self.gam = PoissonGAM(s(0) + s(1) + s(2))

    def time_fit(self):
        # PIRLS loop timing
        self.gam.fit(self.X, self.y)
