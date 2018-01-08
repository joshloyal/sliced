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
    """Sliced Inverse Regression (SIR) [1]

    Linear dimensionality reduction using the inverse regression curve,
    E[X|y], to identify the effective dimension reducing (EDR) directions.
    The inverse comes from the fact that X and y are reversed with respect
    to the standard regression framework (estimating E[y|X]).

    The algorithm performs a weighted principal component analysis on
    slices of the whitened data, which has been sorted with respect to
    the target, y.

    For a binary target the directions found correspond to those found
    with Fisher's Linear Discriminant Analysis (LDA).

    Parameters
    ----------
    n_components : int, None (default=None)
        Number of directions to keep. Corresponds to the dimension of
        the EDR-space.

    n_slices : int (default=10)
        The number of slices used when calculating the inverse regression
        curve. Should be <= the number of unique values of ``y``.

    copy : bool (default=True)
         If False, data passed to fit are overwritten and running
         fit(X).transform(X) will not yield the expected results,
         use fit_transform(X) instead.

    Attributes
    ----------
    components_ : array, shape (n_features, n_components)
        EDR directions in feature space, representing the EDR-subspace
        which is sufficient to describe the conditional distribution
        y|X. The components are sorted by ``eigenvalues_``.

    singular_values_ : array, shape (n_components,)
        The singular values corresponding to each of the selected components.
        These are equivalent to the eigenvalues of the covariance matrix
        of the inverse regression curve. Larger eigenvalues indicate
        more prevelant directions.

    Examples
    --------

    >>> import numpy as np
    >>> from sdr import SlicedInverseRegression
    >>> from sdr.datasets import make_cubic
    >>> X, y = make_cubic(random_state=123)
    >>> X_sir = SlicedInverseRegression(n_components=2).fit_transform(X, y)
    >>> X_sir.shape
    (500, 2)

    References
    ----------

    [1] Li, K-C. (1991) "Sliced Inverse Regression for Dimension Reduction", Journal of
    the American Statistical Association, 86, 316-327.
    """
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
