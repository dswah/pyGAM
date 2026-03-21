"""
Basic Regression Example using pyGAM

This example demonstrates:
1. Generating synthetic nonlinear data
2. Training a GAM regression model
3. Making predictions with confidence intervals
4. Visualizing the results
"""

import matplotlib.pyplot as plt
import numpy as np

from pygam import LinearGAM, s

np.random.seed(0)

X = np.linspace(0, 5, 100)
y = np.sin(X) + np.random.normal(scale=0.2, size=100)


X = X.reshape(-1, 1)


gam = LinearGAM(s(0)).fit(X, y)


X_pred = np.linspace(0, 5, 200).reshape(-1, 1)
y_pred = gam.predict(X_pred)
y_conf = gam.confidence_intervals(X_pred)


#Visualize
plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="Data", alpha=0.6)
# GAM prediction line
plt.plot(X_pred, y_pred, color="red", label="GAM Fit")

# Confidence interval shading
plt.fill_between(
    X_pred.flatten(),
    y_conf[:, 0],
    y_conf[:, 1],
    color="red",
    alpha=0.2,
    label="Confidence Interval"
)

# Labels and title
plt.title("pyGAM Regression Example with Confidence Intervals")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()
