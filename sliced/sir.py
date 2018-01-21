from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import numpy as np
import scipy.linalg as linalg

from scipy import sparse
from sklearn.discriminant_analysis import _cov
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted

from .base import whiten_X, slice_X, is_multioutput


__all__ = ['SlicedInverseRegression']


def grouped_sum(array, groups):
    """Sums an array by groups. Groups are assumed to be contiguous by row."""
    inv_idx = np.concatenate(([0], np.diff(groups).nonzero()[0]))
    return np.add.reduceat(array, inv_idx)


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
    n_components : int, str or None (default='auto')
        Number of directions to keep. Corresponds to the dimension of
        the central subpace. If n_components=='auto', the number of components
        is chosen by finding the maximum gap in the ordered eigenvalues of
        the var(X|y) matrix and choosing the components before this gap.
        If n_components==None, the number of components equals the number of
        features.

    n_slices : int (default=10)
        The number of slices used when calculating the inverse regression
        curve. Should be <= the number of unique values of ``y``.

    copy : bool (default=True)
         If False, data passed to fit are overwritten and running
         fit(X).transform(X) will not yield the expected results,
         use fit_transform(X) instead.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        The directions in feature space, representing the
        central subspace which is sufficient to describe the conditional
        distribution y|X. The components are sorted by ``singular_values_``.

    singular_values_ : array, shape (n_components,)
        The singular values corresponding to each of the selected components.
        These are equivalent to the eigenvalues of the covariance matrix
        of the inverse regression curve. Larger eigenvalues indicate
        more prevelant directions.

    Examples
    --------

    >>> import numpy as np
    >>> from sliced import SlicedInverseRegression
    >>> from sliced.datasets import make_cubic
    >>> X, y = make_cubic(random_state=123)
    >>> sir = SlicedInverseRegression(n_components=2)
    >>> sir.fit(X, y)
    SlicedInverseRegression(copy=True, n_components=2, n_slices=10)
    >>> X_sir = sir.transform(X)

    References
    ----------

    [1] Li, K C. (1991)
        "Sliced Inverse Regression for Dimension Reduction (with discussion)",
        Journal of the American Statistical Association, 86, 316-342.
    """
    def __init__(self, n_components='auto', n_slices=10, copy=True):
        self.n_components = n_components
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
                         accept_sparse=['csr'],
                         y_numeric=True, copy=self.copy)

        # handle n_components == None
        if self.n_components is None:
            self.n_components_ = X.shape[1]
        else:
            self.n_components_ = self.n_components

        # validate y
        if is_multioutput(y):
            raise TypeError("The target `y` cannot be multi-output.")

        # `n_slices` must be less-than or equal to the number of unique values
        # of `y`.
        n_y_values = np.unique(y).shape[0]
        if n_y_values == 1:
            raise ValueError("The target only has one unique y value. It does "
                             "not make sense to fit SIR in this case.")

        if self.n_slices > n_y_values:
            warnings.warn("n_slices greater than number of unique y values. "
                          "Setting n_slices equal to {0}.".format(n_y_values))
            self.n_slices_ = n_y_values
        else:
            self.n_slices_ = self.n_slices

        # Center and Whiten feature matrix using the cholesky decomposition
        # (the original implementation uses QR, but this has numeric errors).
        Z, sigma_inv = whiten_X(X, method='cholesky', copy=False)

        # sort rows of Z with respect to y
        Z = Z[np.argsort(y), :]

        # determine slice indices and counts per slice
        slices, counts = slice_X(Z, self.n_slices_)

        # means in each slice (sqrt factor takes care of the weighting)
        Z_means = grouped_sum(Z, slices) / np.sqrt(counts.reshape(-1, 1))

        # PCA of slice matrix
        U, S, V = linalg.svd(Z_means, full_matrices=True)
        components = np.dot(V, sigma_inv)
        singular_values = (S ** 2)

        # the number of components is chosen by finding the maximum gap among
        # the ordered eigenvalues.
        if self.n_components_ == 'auto':
            self.n_components_ = np.argmax(np.abs(np.diff(singular_values))) + 1

        self.components_ = components[:self.n_components_, :]
        self.singular_values_ = singular_values[:self.n_components_]

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
        X_new : array-like, shape (n_samples, n_components)

        """
        check_is_fitted(self, 'components_')

        X = check_array(X)
        return np.dot(X, self.components_.T)
