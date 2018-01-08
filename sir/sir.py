import numpy as np

from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array


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

        X = check_array(X, dtype=[np.float64, np.float32], ensure_2d=True,
                        copy=self.copy)

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
        slices = np.repeat(np.arange(1, self.n_slices),
                           np.ceil(n_samples / self.n_slices))[:n_samples]
        slice_counts = np.bincount(slices)[1:]

        # means in each slice (takes care of the weighting)
        Z_sliced = grouped_sum(Z, slices) / np.sqrt(slice_counts.reshape(-1,1))

        # PCA of slice matrix
        U, S, V = linalg.svd(Z_sliced, full_matrices=True)
        self.components_ = np.dot(V.T, L_inv)[:, :self.n_components]
        self.singular_values_ = S ** 2

        return self

    def transform(self, X):
        return np.dot(X, self.components_)
