import numpy as np

from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_X_y


class SlicedAverageVarianceEstimation(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, n_slices=10, copy=True):
        self.n_components = n_components
        self.n_slices = n_slices
        self.copy = copy

    def fit(self, X, y):
        n_samples, n_features = X.shape

        X, y = check_X_y(X, y, dtype=[np.float64, np.float32],
                         y_numeric=True, copy=self.copy)

        # center data
        X -= np.mean(X, axis=0)

        # whiten data using cholesky decomposition
        sigma = np.dot(X.T, X) / (n_samples - 1)
        L = linalg.cholesky(sigma)
        L_inv = linalg.solve_triangular(L, np.eye(L.shape[0]))
        Z = np.dot(X, L_inv)

        # sort rows of Z w.r.t y
        Z = Z[np.argsort(y), :]

        # determine slices and counts
        slices = np.repeat(np.arange(self.n_slices),
                           np.ceil(n_samples / self.n_slices))[:n_samples]
        slice_counts = np.bincount(slices)

        # construct slice covariance matrices
        Z_sliced = np.zeros((n_features, n_features))
        for slice_idx in range(self.n_slices):
            n_slice = slice_counts[slice_idx]
            Z_slice = Z[slices == slice_idx, :]
            Z_slice = Z_slice - np.mean(Z_slice, axis=0)
            V_slice = np.eye(n_features) - (np.dot(Z_slice.T, Z_slice) / (n_slice - 1))
            Z_sliced += (n_slice / n_samples) * np.dot(V_slice, V_slice)

        # PCA of slice matrix
        U, S, V = linalg.svd(Z_sliced, full_matrices=True)
        self.components_ = np.dot(V.T, L_inv)[:, :self.n_components]
        self.singular_values_ = S ** 2

        return self

    def transform(self, X):
        return np.dot(X, self.components_)
