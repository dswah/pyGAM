import numpy as np
from pygam import LinearGAM

X = np.random.rand(100, 1)
y = np.sin(X[:, 0])

gam = LinearGAM().fit(X, y)

print("Model trained successfully\n")

try:
    result = gam.predict([[0.5], [0.2]])
    print("Test 1 Passed: Valid input works")
except Exception as e:
    print("Test 1 Failed:", e)

try:
    result = gam.predict([0.5, 0.2])
    print("Test 2 Passed: 1D input handled")
except Exception as e:
    print("Test 2 Failed:", e)

try:
    gam.predict([[1, 2], [3, 4]])
    print("Test 3 Failed: Should have raised error")
except ValueError as e:
    print("Test 3 Passed:", e)

try:
    gam.predict([[0.5], [np.nan]])
    print("Test 4 Failed: Should have raised error")
except ValueError as e:
    print("Test 4 Passed:", e)