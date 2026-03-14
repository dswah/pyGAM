from pygam import LinearGAM, s
import numpy as np
import warnings

# Suppress warnings to catch them manually if needed
# warnings.filterwarnings("ignore")

print("--- Running Identifiability Reproduction Script ---")

# Simulate data
np.random.seed(42)
X = np.linspace(0, 1, 100).reshape(-1, 1)
y = np.sin(2 * np.pi * X).flatten() + np.random.normal(0, 0.1, 100)

# Fit model with intercept and spline
print("Fitting LinearGAM(s(0)) with default intercept...")
gam = LinearGAM(s(0)).fit(X, y)

print("\nModel Summary:")
gam.summary()

# Check for the warning in the summary
print("\n--- End of Reproduction ---")
