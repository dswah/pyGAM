import numpy as np


class EDoFBenchmark:
    """
    Benchmarks the O(N^3) memory bottleneck in EDoF calculation.
    Targeting ~800MB spike to stay safe within 7GB CI limits.
    """

    timeout = 120
    N_FEATURES = 10000
    N_SAMPLES = 500

    def setup(self):
        np.random.seed(42)
        self.U1 = np.random.rand(self.N_FEATURES, self.N_SAMPLES)

    def time_legacy_edof(self):
        return np.diagonal(self.U1.dot(self.U1.T))

    def peakmem_legacy_edof(self):
        return np.diagonal(self.U1.dot(self.U1.T))
