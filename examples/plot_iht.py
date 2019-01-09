"""
================================
Iterative Hessian Transformation
================================

An example plot of :class:`sliced.iht.IterativeHessianTransformation`
applied to the Penn Digits Dataset.
"""
import numpy as np
import matplotlib.pyplot as plt

from sliced import IterativeHessianTransformation
from sliced.datasets import load_penn_digits

X_train, y_train = load_penn_digits(subset='train', digits=[0, 6, 9])
X_test, y_test = load_penn_digits(subset='test', digits=[0, 6, 9])


iht = IterativeHessianTransformation(target_type='response').fit(X_train, y_train)
X_iht = iht.transform(X_test)

# plot data projected onto the first direction
plt.scatter(X_iht[:, 0], X_iht[:, 1], c=y_test, linewidth=0.5, edgecolor='k')
plt.xlabel("$X\hat{\\beta_1}$")
plt.ylabel("$X\hat{\\beta_2}$")

plt.show()
