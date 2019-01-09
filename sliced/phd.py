from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import numpy as np
import scipy.linalg as linalg

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import normalize, StandardScaler

from .externals import check_array, check_X_y, check_is_fitted
from .base import is_multioutput


__all__ = ['PrincipalHessianDirections']


class PrincipalHessianDirections(BaseEstimator, TransformerMixin):
    """Principal Hessian Directions (PHD) [1]

    The Principal Hessian Directions (PHD) method of Li (1992) estimates the span of
    the central mean subspace S_{E[y|X]}. This in attractive when one is primarly
    interested in  estimating the condtional mean E[X|Y] as opposed to full conditional
    distribution F_{X|Y}. For example in machine learning and non-parametric regression
    where one is interested in the model:
                    Y = m(X) + epsilon
    where m(X) is the regression function, E[X|Y].

    The algorithm performs an eigen-decomposition of the the third-moment E[XX^T Y].
    This estimator also supports the residual based method suggested by Li, which uses
    the estimate E[XX^T r] where r is the residual of the least squares fit of y on X.
    Note that PHD reguires the linear conditional moment and concstant conditional
    variance condition to hold.

    Parameters
    ----------
    n_directions : int, str or None (default='auto')
        Number of directions to keep. Corresponds to the dimension of
        the central mean subpace. If n_directions==None,
        the number of directions equals the number of features.

    target_type : str, 'response' or 'residual', (default='residuals')
        Transformation of the target to use when estimating the third moment.
        The default is to use the residual based E[XX^T r] as opposed to the response
        based moment E[XX^T y].

    copy : bool (default=True)
         If False, data passed to fit are overwritten and running
         fit(X).transform(X) will not yield the expected results,
         use fit_transform(X) instead.

    Attributes
    ----------
    directions_ : array, shape (n_directions, n_features)
        The directions in feature space, representing the
        central mean subspace which is sufficient to describe the conditional
        mean E[y|X]. The directions are sorted by ``eigenvalues_``.

    eigenvalues_ : array, shape (n_directions,)
        The eigenvalues corresponding to each of the selected directions.
        These are the eigenvalues of the expected outer product of gradients.
        Larger eigenvalues indicate more prevalent directions.

    mean_ : array, shape (n_features,)
        The column means of the training data used to estimate the basis
        of the central mean subspace. Used to project new data onto the
        central mean subspace.

    References
    ----------

    [1] Ker-Chau Li (1992),
        "On Principal Hessian Directions for Data Visualization and Dimension Reduction:
        Another Application of Stein's Lemma",
        Journal of the American Statistical Society, 1992 vol. 87 (420) pp. 1025-1039.
    """
    def __init__(self, n_directions=None, target_type="residuals", copy=True):
        self.n_directions = n_directions
        self.target_type = target_type
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
            raise TypeError("PrincipalHessianDirections does not support "
                            "sparse input.")

        X, y = check_X_y(X, y, dtype=[np.float64, np.float32],
                         y_numeric=True, copy=self.copy)
        n_samples, n_features = X.shape

        if self.target_type not in {'response', 'residual'}:
            raise ValueError("`target_type` must be one of {'response', 'residual'}")

        self.n_directions_ = (n_features if self.n_directions is None else
                              self.n_directions)

        # validate y
        if is_multioutput(y):
            raise TypeError("The target `y` cannot be multi-output.")

        # transform the target for the computation of sigma_{yxx}
        if self.target_type == 'response':
            y = y - np.mean(y)
        else:
            linear_model = LinearRegression(fit_intercept=True).fit(X, y)
            y = y - linear_model.predict(X)

        # Center and Whiten feature matrix using a QR decomposition
        # (this is the approach used in the dr package)
        self.mean_ = np.mean(X, axis=0)
        if self.copy:
            X = X - self.mean_
        else:
            X -= self.mean_
        Q, R = linalg.qr(X, mode='economic')
        Z = np.sqrt(n_samples) * Q

        # calculate sigma_{yzz}
        Zy = Z * y.reshape(-1, 1)
        M = np.dot(Zy.T, Z) / n_samples

        # eigen-decomposition to determine the basis
        evals, evecs = linalg.eigh(M)

        # re-order eigenvectors based on magnitude
        # NOTE: M is not psd, so eigenvalues may be negative.
        order = np.argsort(np.abs(evals))[::-1]
        evecs = evecs[:, order]
        evals = evals[order]

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

        directions = normalize(
            directions[:, :self.n_directions_], norm='l2', axis=0)
        self.directions_ = directions.T
        self.eigenvalues_ = evals[:self.n_directions_]

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
        return np.dot(X - self.mean_, self.directions_.T)
