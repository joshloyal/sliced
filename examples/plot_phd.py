"""
==================================
Principal Hessian Directions (PHD)
==================================

An example plot of :class:`sliced.phd.PrincipalHessianDirections`
"""
import numpy as np
import matplotlib.pyplot as plt

from sliced import PrincipalHessianDirections
from sliced import datasets


from sliced.datasets import load_athletes

X, y = load_athletes()

phd = PrincipalHessianDirections(target_type='residual')
X_phd = phd.fit_transform(X, y)

# plot data projected onto the first direction
plt.scatter(X_phd[:, 0], y, c=y, cmap='viridis', linewidth=0.5, edgecolor='k')
plt.xlabel("$X\hat{\\beta_1}$")
plt.ylabel("y")

plt.show()
