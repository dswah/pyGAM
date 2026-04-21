import os

# Lock threads for deterministic memory and time profiling
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np


class EDoFBenchmark:
    """
    Micro-benchmarks for Effective Degrees of Freedom (EDoF) calculation.
    Compares legacy O(N^3) memory bottleneck vs optimized O(N) approach.
    """

    timeout = 120
    # Dimensions chosen to safely hit ~800MB RAM, well below CI 7GB limit
    N_FEATURES = 10000
    N_SAMPLES = 500

    def setup(self):
        np.random.seed(42)
        self.U1 = np.random.rand(self.N_FEATURES, self.N_SAMPLES)

    def time_legacy_edof(self):
        # Legacy dense matrix multiplication O(N^3) time
        return np.diagonal(self.U1.dot(self.U1.T))

    def peakmem_legacy_edof(self):
        # Legacy dense matrix multiplication O(N^2) space (~800MB)
        return np.diagonal(self.U1.dot(self.U1.T))

    def time_optimized_edof(self):
        # Optimized vectorized calculation O(N) time
        return (self.U1**2).sum(axis=1)

    def peakmem_optimized_edof(self):
        # Optimized vectorized calculation O(1) extra space
        return (self.U1**2).sum(axis=1)
