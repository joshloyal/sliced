import numpy as np

from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_X_y


from .base import whiten_X, slice_X


class SlicedAverageVarianceEstimation(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, n_slices=10, copy=True):
        self.n_components = n_components
        self.n_slices = n_slices
        self.copy = copy

    def fit(self, X, y):
        n_samples, n_features = X.shape

        X, y = check_X_y(X, y, dtype=[np.float64, np.float32],
                         y_numeric=True, copy=self.copy)

        # center and whiten  data
        Z, sigma_inv = whiten_X(X)

        # sort rows of Z with respect to the target y
        Z = Z[np.argsort(y), :]

        # determine slices and counts
        slices, counts = slice_X(Z, self.n_slices)

        # construct slice covariance matrices
        M = np.zeros((n_features, n_features))
        for slice_idx in range(self.n_slices):
            n_slice = counts[slice_idx]

            # center the entries in this slice
            Z_slice = Z[slices == slice_idx, :]
            Z_slice -= np.mean(Z_slice, axis=0)

            # slice covariance matrix
            V_slice = np.dot(Z_slice.T, Z_slice) / (n_slice - 1)
            M_slice = np.eye(n_features) - V_slice
            M += (n_slice / n_samples) * np.dot(M_slice, M_slice)

        # PCA of slice matrix
        U, S, V = linalg.svd(M, full_matrices=True)
        self.components_ = np.dot(V.T, sigma_inv)[:, :self.n_components]
        self.singular_values_ = S ** 2

        return self

    def transform(self, X):
        return np.dot(X, self.components_)
