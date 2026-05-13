import gc
import os
import threading
import time

import numpy as np
import psutil
import pytest

from pygam import LinearGAM, s

# --- Configuration ---
# LEAK TOLERANCE: How much extra memory (MB) can be left over?
# We allow a small buffer.
LEAK_TOLERANCE_MB = 50

# PEAK THRESHOLD: If memory usage exceeds 'X' times the starting memory, fail.
PEAK_MEMORY_MULTIPLIER = 4.0


class PeakMemoryMonitor:
    def __init__(self, pid):
        self.process = psutil.Process(pid)
        self.stop_event = threading.Event()
        self.peak_memory = 0.0
        self.start_memory = 0.0
        self.thread = threading.Thread(target=self._monitor)

    def _monitor(self):
        while not self.stop_event.is_set():
            # Get current memory in MB
            current_mem = self.process.memory_info().rss / 1024 / 1024
            print(f"Current Memory: {current_mem:.2f} MB")

            if current_mem > self.peak_memory:
                self.peak_memory = current_mem

            if self.peak_memory > self.start_memory * PEAK_MEMORY_MULTIPLIER:
                pytest.fail(
                    f"Peak memory usage ({self.peak_memory:.2f} MB) exceeded safety limit"
                )

            time.sleep(0.5)

    def start(self):
        self.start_memory = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = self.start_memory
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()


def test_gridsearch_memory_robust():
    """
    Robustly tests for memory leaks by comparing Start vs. End memory
    and ensuring peak memory doesn't explode.
    """
    N_SAMPLES = 1000
    N_GRID_POINTS = 100

    rng = np.random.default_rng(42)
    X = rng.standard_normal((N_SAMPLES, 10))
    y = np.sin(X[:, 0]) + X[:, 1] ** 2 + rng.standard_normal(N_SAMPLES) * 0.1

    gam = LinearGAM(s(0) + s(1) + s(2))
    lam_grid = np.logspace(-3, 3, N_GRID_POINTS)

    # Force GC to ensure we start at a true baseline
    gc.collect()
    time.sleep(0.5)

    monitor = PeakMemoryMonitor(os.getpid())
    monitor.start()

    print(f"\n[Test Info] Starting Memory: {monitor.start_memory:.2f} MB")

    try:
        gam.gridsearch(X, y, lam=lam_grid, progress=False)
    except Exception as e:
        pytest.fail(f"Gridsearch failed with error: {e}")
    finally:
        monitor.stop()

    gc.collect()

    process = psutil.Process(os.getpid())
    end_memory = process.memory_info().rss / 1024 / 1024

    print(f"[Test Info] Peak Memory:     {monitor.peak_memory:.2f} MB")
    print(f"[Test Info] Ending Memory:   {end_memory:.2f} MB")

    max_allowed_peak = monitor.start_memory * PEAK_MEMORY_MULTIPLIER
    if monitor.peak_memory > max_allowed_peak:
        pytest.fail(
            f"Peak memory usage ({monitor.peak_memory:.2f} MB) exceeded safety limit "
            f"({max_allowed_peak:.2f} MB). This suggests runaway memory usage."
        )

    diff = end_memory - monitor.start_memory

    if diff > LEAK_TOLERANCE_MB:
        pytest.fail(
            f"Memory Leak Detected! Memory grew by {diff:.2f} MB (Threshold: {LEAK_TOLERANCE_MB} MB). "
            f"Start: {monitor.start_memory:.2f} MB, End: {end_memory:.2f} MB."
        )

    print(f"[Test Info] Memory Difference: {diff:.2f} MB (PASSED)")


if __name__ == "__main__":
    test_gridsearch_memory_robust()
    print("Test finished successfully.")
