"""
=======================
Binary Targets with SIR
=======================

Sliced Inverse Regression is able to find a one-dimensional subspace
that seperates cases in the famous breast cancer dataset.
"""
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sliced import SlicedInverseRegression

X, y = load_breast_cancer(return_X_y=True)

sir = SlicedInverseRegression(n_directions=2).fit(X, y)
X_sir = sir.transform(X)

plt.scatter(X_sir[:, 0], X_sir[:, 1], c=y, alpha=0.8, edgecolor='k')
plt.xlabel("$X\hat{\\beta}_{1}$")
plt.ylabel("$X\hat{\\beta}_{2}$")
plt.title("Breast Cancer Data")

plt.show()
