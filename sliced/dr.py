from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import numpy as np
import scipy.linalg as linalg

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize

from .externals import check_array, check_X_y, check_is_fitted
from .base import slice_y, is_multioutput


__all__ = ['DirectionalRegression']


class DirectionalRegression(BaseEstimator, TransformerMixin):
    """Directional Regression (DR)"""
    def __init__(self, n_directions=None, n_slices=10, copy=True):
        self.n_directions = n_directions
        self.n_slices = n_slices
        self.copy = copy

    def fit(self, X, y):
        """Fit the model with X and y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target values (class labels in classification, real numbers
            in regression).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if sparse.issparse(X):
            raise TypeError("DirectionalRegression does not support "
                            "sparse input.")

        X, y = check_X_y(X, y, dtype=[np.float64, np.float32],
                         accept_sparse=['csr'],
                         y_numeric=True, copy=self.copy)

        if self.n_directions is None:
            n_directions = X.shape[1]
        elif (not isinstance(self.n_directions, six.string_types) and
                self.n_directions < 1):
            raise ValueError('The number of directions `n_directions` '
                             'must be >= 1. Got `n_directions`={}'.format(
                                self.n_directions))
        else:
            n_directions = self.n_directions

        # validate y
        if is_multioutput(y):
            raise TypeError("The target `y` cannot be multi-output.")

        n_samples, n_features = X.shape

        # Center and Whiten feature matrix using a QR decomposition
        # (this is the approach used in the dr package)
        self.mean_ = np.mean(X, axis=0)
        if self.copy:
            X = X - self.mean_
        else:
            X -= self.mean_
        Q, R = linalg.qr(X, mode='economic')
        Z = np.sqrt(n_samples) * Q
        Z = Z[np.argsort(y), :]

        # determine slice indices and counts per slice
        slices, counts = slice_y(y, self.n_slices)
        self.n_slices_ = counts.shape[0]

        # Construct moment matrices
        Mz = np.zeros((n_features, n_features))
        Mzzt = np.zeros((n_features, n_features))
        for slice_idx in range(self.n_slices_):
            n_slice = counts[slice_idx]
            p_slice = n_samples / n_slice

            # extract entries in the slice
            Z_slice = Z[slices == slice_idx, :]

            # extract empirical moments
            Ez = np.mean(Z_slice, axis=0).reshape(-1, 1)
            Ezzt = np.dot(Z_slice.T, Z_slice) / n_slice

            # first moment matrix
            Mz += p_slice * np.dot(Ez, Ez.T)

            # second moment matrix
            Mzzt += p_slice * np.dot(Ezzt, Ezzt)

        M = 2 * Mzzt + 2 * np.dot(Mz, Mz) + 2 * np.trace(Mz) * Mz
        M -= 2 * np.eye(n_features)

        # eigen-decomposition of slice matrix
        evals, evecs = linalg.eigh(M)
        evecs = evecs[:, ::-1]
        evals = evals[::-1]
        try:
            # TODO: internally handle zero variance features. This would not
            # be a problem if we used svd, but does not match DR.
            directions = linalg.solve_triangular(np.sqrt(n_samples) * R, evecs)
        except (linalg.LinAlgError, TypeError):
            # NOTE: The TypeError is because of a bug in the reporting of scipy
            raise linalg.LinAlgError(
                "Unable to back-solve R for the dimension "
                "reducing directions. This is usually caused by the presents "
                "of zero variance features. Try removing these features with "
                "`sklearn.feature_selection.VarianceThreshold(threshold=0.)` "
                "and refitting.")

        # the number of directions is chosen by finding the maximum gap among
        # the ordered eigenvalues.
        if self.n_directions == 'auto':
            n_directions = np.argmax(np.abs(np.diff(evals))) + 1
        self.n_directions_ = n_directions

        # normalize directions
        directions = normalize(
            directions[:, :self.n_directions_], norm='l2', axis=0)
        self.directions_ = directions.T

        self.eigenvalues_ = evals[:self.n_directions_]
        self.eigenvectors_ = evecs[:, :self.n_directions_].T

        return self

    def transform(self, X):
        """Apply dimension reduction on X.

        X is projected onto the EDR-directions previously extracted from a
        training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.


        Returns
        -------
        X_new : array-like, shape (n_samples, n_directions)

        """
        check_is_fitted(self, 'directions_')

        X = check_array(X)
        return np.dot(X - self.mean_, self.directions_.T)
