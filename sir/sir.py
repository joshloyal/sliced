import numpy as np

from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array


def grouped_sum(array, groups, axis=0, issorted=False):
    array = np.asarray(array)
    groups = np.asarray(groups)

    if issorted:
        aux = groups
        ordered_array = array
    else:
        perm = groups.argsort()
        aux = groups[perm]
        ordered_array = array[perm]

    flag = np.concatenate(([True], aux[1:] != aux[:-1]))
    inv_idx, = flag.nonzero()

    result = np.add.reduceat(ordered_array, inv_idx)

    return result


class SlicedInverseRegression(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, n_slices=10, copy=True):
        self.n_components = n_components
        self.n_slices = n_slices
        self.copy = copy

    def fit(self, X, y):
        n_samples, n_features = X.shape

        X = check_array(X, dtype=[np.float64, np.float32], ensure_2d=True,
                        copy=self.copy)

        # center adata
        X -= np.mean(X, axis=0)

        # whiten data using cholesky decomposition
        sigma = np.cov(X.T)
        L = linalg.cholesky(sigma)
        L_inv = linalg.solve_triangular(L, np.eye(L.shape[0]))
        Z = np.dot(X, L_inv)

        # sort rows of Z w.r.t y
        Z = Z[np.argsort(y), :]

        # determine slice indices
        slices = np.repeat(np.arange(1, self.n_slices),
                           np.ceil(n_samples / self.n_slices))[:n_samples]
        slice_counts = np.bincount(slices)[1:]

        # means in each slice
        Z_sliced = grouped_sum(Z, slices) / slice_counts.reshape(-1, 1)

        # covariance
        Z_sliced *= np.sqrt(slice_counts).reshape(-1, 1)
        Z_cov = np.dot(Z_sliced.T, Z_sliced)

        # eigen decomposition
        evalues, evectors = linalg.eig(Z_cov)
        self.vectors_ = np.dot(evectors, L_inv)
        self.vectors_ = self.vectors_[:, :self.n_components]

        return self

    def transform(self, X):
        return np.dot(X, self.vectors_)
