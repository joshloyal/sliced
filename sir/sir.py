import abc
import six

import numpy as np

from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_X_y


from .base import whiten_X, slice_X


def grouped_sum(array, groups):
    inv_idx = np.concatenate(([0], np.diff(groups).nonzero()[0]))
    return np.add.reduceat(array, inv_idx)


class SlicedInverseRegression(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, n_slices=10, copy=True):
        self.n_components = n_components
        self.n_slices = n_slices
        self.copy = copy

    def fit(self, X, y):
        n_samples, n_features = X.shape

        X, y = check_X_y(X, y, dtype=[np.float64, np.float32],
                         y_numeric=True, copy=self.copy)

        # center and whiten feature matrix
        Z, sigma_inv = whiten_X(X)

        # sort rows of Z with respect to y
        Z = Z[np.argsort(y), :]

        # determine slice indices and counts per slice
        slices, counts = slice_X(Z, self.n_slices)

        # means in each slice (sqrt factor takes care of the weighting)
        Z_means = grouped_sum(Z, slices) / np.sqrt(counts.reshape(-1,1))

        # PCA of slice matrix
        U, S, V = linalg.svd(Z_means, full_matrices=True)
        self.components_ = np.dot(V.T, sigma_inv)[:, :self.n_components]
        self.singular_values_ = S ** 2

        return self

    def transform(self, X):
        return np.dot(X, self.components_)
