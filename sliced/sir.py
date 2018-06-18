from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import warnings

import numpy as np
import scipy.linalg as linalg
import scipy.stats as stats

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize

from .externals import check_array, check_X_y, check_is_fitted
from .base import slice_y, grouped_sum, is_multioutput


__all__ = ['SlicedInverseRegression']


def diagonal_precision(X, R=None):
    """Calculate he diagonal elements of the empirical precision matrix
    from a pre-computed QR decomposition of the centered data matrix X.

    Parameters
    ----------
    X : numpy-array, shape (n_samples, n_features)
        The centered data matrix.

    R : numpy-array, shape (n_features, n_features), optional
        The upper triangular matrix obtained by the economic QR decomposition
        of the centered data matrix.

    Returns
    -------
    precisions : numpy-array, shape (n_features,)
        The diagonal elements of the precision (inverse covariance) matrix of
        the centered data.
    """
    if R is None:
        Q, R = linalg.qr(X, mode='economic')

    R_inv = linalg.solve_triangular(R, np.eye(R.shape[1]))
    return np.sum(R_inv ** 2, axis=1) * X.shape[0]


class SlicedInverseRegression(BaseEstimator, TransformerMixin):
    """Sliced Inverse Regression (SIR) [1]

    Linear dimensionality reduction using the inverse regression curve,
    E[X|y], to identify the directions defining the central subspace of
    the data.

    The inverse comes from the fact that X and y are reversed with respect
    to the standard regression framework (estimating E[y|X]).

    The algorithm performs a weighted principal component analysis on
    slices of the whitened data, which has been sorted with respect to
    the target, y.

    For a binary target the directions found correspond to those found
    with Fisher's Linear Discriminant Analysis (LDA).

    Note that SIR may fail to estimate the directions if the conditional
    density X|y is symmetric, so that E[X|y] = 0. See
    :class:`sliced.save.SlicedAverageVarianceEstimation`,
    which is able to overcome this limitation but may fail to pick up on
    linear trends. If possible, both SIR and SAVE should be used when analyzing
    a dataset.

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

    alpha : float or None (default=None)
        Significance level for the two-sided t-test used to check for
        non-zero coefficients. Must be a number between 0 and 1. If not
        `None`, the non-zero components of each direction are determined
        from an asymptotic normal approximation. Useful if one desires that
        the directions are sparse in the number of features.

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
        more prevalent directions.

    Examples
    --------

    >>> import numpy as np
    >>> from sliced import SlicedInverseRegression
    >>> from sliced.datasets import make_cubic
    >>> X, y = make_cubic(random_state=123)
    >>> sir = SlicedInverseRegression(n_directions=2)
    >>> sir.fit(X, y)
    SlicedInverseRegression(alpha=None, copy=True, n_directions=2, n_slices=10)
    >>> X_sir = sir.transform(X)

    References
    ----------

    [1] Li, K C. (1991)
        "Sliced Inverse Regression for Dimension Reduction (with discussion)",
        Journal of the American Statistical Association, 86, 316-342.
    [2] Chen, C.H., and Li, K.C. (1998), "Can SIR Be as Popular as Multiple
        Linear Regression?" Statistica Sinica, 8, 289-316.
    """
    def __init__(self, n_directions='auto', n_slices=10, alpha=None, copy=True):
        self.n_directions = n_directions
        self.n_slices = n_slices
        self.alpha = alpha
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

        if self.alpha is not None and (self.alpha <= 0 or self.alpha >= 1):
            raise ValueError("The significance level `alpha` "
                             "must be between 0 and 1. Got `alpha`={}".format(
                                self.alpha))

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
        Z = Z[np.argsort(y), :]

        # determine slice indices and counts per slice
        slices, counts = slice_y(y, self.n_slices)
        self.n_slices_ = counts.shape[0]

        # means in each slice (sqrt factor takes care of the weighting)
        Z_means = grouped_sum(Z, slices) / np.sqrt(counts.reshape(-1, 1))
        M = np.dot(Z_means.T, Z_means) / n_samples

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

        directions = normalize(
            directions[:, :self.n_directions_], norm='l2', axis=0)
        self.directions_ = directions.T
        self.eigenvalues_ = evals[:self.n_directions_]

        # Drop components in each direction using the t-ratio approach
        # suggested in section 4 of Chen and Li (1998).
        if self.alpha is not None:
            # similar to multiple linear-regression the standard error
            # estimates are proportional to the diagonals of the inverse
            # covariance matrix.
            precs = diagonal_precision(X, R=R)
            weights = (1 - self.eigenvalues_) / (n_samples * self.eigenvalues_)
            std_error = np.sqrt(weights.reshape(-1, 1) * precs)

            # perform a two-sided t-test at level alpha for coefs equal to zero
            # NOTE: we are not correcting for multiple tests and this is
            #       a very rough approximation, so do not expect the test
            #       to be close to the nominal level.
            df = n_samples - n_features - 1
            crit_val = stats.distributions.t.ppf(1 - self.alpha/2, df)
            for j in range(self.n_directions_):
                test_stat = np.abs(self.directions_[j, :] / std_error[j])
                zero_mask = test_stat < crit_val
                if np.sum(zero_mask) == n_features:
                    warnings.warn("Not zeroing out coefficients. All "
                                  "coefficients are not significantly "
                                  "different from zero.", RuntimeWarning)
                else:
                    self.directions_[j, test_stat < crit_val] = 0.

        return self

    def transform(self, X):
        """Apply dimension reduction on X.

        X is projected onto the central subspace  previously extracted from a
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
