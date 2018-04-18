from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import warnings

import numpy as np
import scipy.linalg as linalg

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import normalize

from .base import slice_y, is_multioutput


__all__ = ['SlicedAverageVarianceEstimation']


class SlicedAverageVarianceEstimation(BaseEstimator, TransformerMixin):
    """Sliced Average Variance Estimation (SAVE) [1]

    Linear dimensionality reduction using the conditional covariance, Cov(X|y),
    to identify the directions defining the central subspace of the data.

    The algorithm performs a weighted principal component analysis on a
    transformation of slices of the covariance matrix of the whitened
    data, which has been sorted with respect to the target, y.

    Since SAVE looks at second moment information, it may miss first-moment
    information. In particular, it may miss linear trends. See
    :class:`sliced.sir.SlicedInverseRegression`, which is able to detect
    linear trends but may fail in other situations. If possible, both SIR and
    SAVE should be used when analyzing a dataset.

    Parameters
    ----------
    n_directions : int, str or None (default='auto')
        Number of directions to keep. Corresponds to the dimension of
        the central subpace. If n_directions=='auto', the number of directions
        is chosen by finding the maximum gap in the ordered eigenvalues of
        the var(X|y) matrix and choosing the directions before this gap.
        If n_directions==None, the number of directions equals the number of
        features.

    n_slices : int (default=10)
        The number of slices used when calculating the inverse regression
        curve. Truncated to at most the number of unique values of ``y``.

    copy : bool (default=True)
         If False, data passed to fit are overwritten and running
         fit(X).transform(X) will not yield the expected results,
         use fit_transform(X) instead.

    Attributes
    ----------
    directions_ : array, shape (n_directions, n_features)
        The directions in feature space, representing the
        central subspace which is sufficient to describe the conditional
        distribution y|X. The directions are sorted by ``eigenvalues_``.

    eigenvalues_ : array, shape (n_directions,)
        The eigenvalues corresponding to each of the selected directions.
        These are the eigenvalues of the covariance matrix
        of the inverse regression curve. Larger eigenvalues indicate
        more prevelant directions.

    Examples
    --------

    >>> import numpy as np
    >>> from sliced import SlicedAverageVarianceEstimation
    >>> from sliced.datasets import make_quadratic
    >>> X, y = make_quadratic(random_state=123)
    >>> save = SlicedAverageVarianceEstimation(n_directions=2)
    >>> save.fit(X, y)
    SlicedAverageVarianceEstimation(copy=True, n_directions=2, n_slices=10)
    >>> X_save = save.transform(X)

    References
    ----------

    [1] Shao, Y, Cook, RD and Weisberg, S (2007).
        "Marginal Tests with Sliced Average Variance Estimation",
        Biometrika, 94, 285-296.
    """
    def __init__(self, n_directions='auto', n_slices=10, copy=True):
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
            raise TypeError("SlicedInverseRegression does not support "
                            "sparse input.")

        X, y = check_X_y(X, y, dtype=[np.float64, np.float32],
                         y_numeric=True, copy=self.copy)

        # handle n_directions == None
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
        if self.copy:
            X = X - np.mean(X, axis=0)
        else:
            X -= np.mean(X, axis=0)
        Q, R = linalg.qr(X, mode='economic')
        Z = np.sqrt(n_samples) * Q

        # sort rows of Z with respect to the target y
        Z = Z[np.argsort(y), :]

        # determine slices and counts
        slices, counts = slice_y(y, self.n_slices)
        self.n_slices_ = counts.shape[0]

        # construct slice covariance matrices
        M = np.zeros((n_features, n_features))
        for slice_idx in range(self.n_slices_):
            n_slice = counts[slice_idx]

            # center the entries in this slice
            Z_slice = Z[slices == slice_idx, :]
            Z_slice -= np.mean(Z_slice, axis=0)

            # slice covariance matrix
            V_slice = np.dot(Z_slice.T, Z_slice) / n_slice
            M_slice = np.eye(n_features) - V_slice
            M += (n_slice / n_samples) * np.dot(M_slice, M_slice)

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
        return np.dot(X, self.directions_.T)
