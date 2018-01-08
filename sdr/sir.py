import numpy as np
import scipy.linalg as linalg

from scipy.sparse import issparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted

from .base import whiten_X, slice_X, is_multioutput


def grouped_sum(array, groups):
    inv_idx = np.concatenate(([0], np.diff(groups).nonzero()[0]))
    return np.add.reduceat(array, inv_idx)


class SlicedInverseRegression(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, n_slices=10, copy=True):
        self.n_components = n_components
        self.n_slices = n_slices
        self.copy = copy

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=[np.float64, np.float32],
                         y_numeric=True, copy=self.copy)

        # handle n_components == None
        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components

        # validate y
        if is_multioutput(y):
            raise TypeError("The target `y` cannot be multi-output.")

        # `n_slices` must be less-than or equal to the number of unique values
        # of `y`.
        n_y_values = np.unique(y).shape[0]
        if self.n_slices > n_y_values:
            n_slices = n_y_values
        else:
            n_slices = self.n_slices

        # Center and Whiten feature matrix using the cholesky decomposition
        # (the original implementation uses QR, but this has numeric errors).
        Z, sigma_inv = whiten_X(X, method='cholesky', copy=False)

        # sort rows of Z with respect to y
        Z = Z[np.argsort(y), :]

        # determine slice indices and counts per slice
        slices, counts = slice_X(Z, n_slices)

        # means in each slice (sqrt factor takes care of the weighting)
        Z_means = grouped_sum(Z, slices) / np.sqrt(counts.reshape(-1,1))

        # PCA of slice matrix
        U, S, V = linalg.svd(Z_means, full_matrices=True)
        self.components_ = np.dot(V.T, sigma_inv)[:, :n_components]
        self.singular_values_ = S ** 2

        return self

    def transform(self, X):
        check_is_fitted(self, 'components_')

        X = check_array(X)
        return np.dot(X, self.components_)
