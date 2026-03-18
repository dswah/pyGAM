"""
Basic Regression Example using pyGAM

This example demonstrates:
1. Generating synthetic nonlinear data
2. Training a GAM regression model
3. Making predictions
4. Visualizing the results
"""

import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s

# Compatible with pyGAM >= 0.12.0

# -------------------------------
# Step 1: Generate sample data
# -------------------------------
np.random.seed(0)
X = np.linspace(0, 5, 100)
y = np.sin(X) + np.random.normal(scale=0.2, size=100)

X = X.reshape(-1, 1)

# -------------------------------
# Step 2: Train the model
# -------------------------------
gam = LinearGAM(s(0)).fit(X, y)

# -------------------------------
# Step 3: Make predictions
# -------------------------------
X_pred = np.linspace(0, 5, 200).reshape(-1, 1)
y_pred = gam.predict(X_pred)

# -------------------------------
# Step 4: Visualize results
# -------------------------------
plt.scatter(X, y, label="Data", alpha=0.6)
plt.plot(X_pred, y_pred, color="red", label="GAM Fit")
plt.title("pyGAM Regression Example")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()