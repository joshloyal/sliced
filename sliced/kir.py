from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import numpy as np
import scipy.linalg as linalg

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import normalize

from .externals import check_array, check_X_y, check_is_fitted
from .base import is_multioutput


__all__ = ['KernelInverseRegression']


class KernelInverseRegression(BaseEstimator, TransformerMixin):
    """Kernel Inverse Regression [1]

    Linear dimensionality reduction using the inverse regression curve,
    E[X|y], to identify the directions defining the central subspace of
    the data. The inverse comes from the fact that X and y are reversed with respect
    to the standard regression framework (estimating E[y|X]).

    Sliced Inverse Regression estimates the inverse regression curve through
    binning the response and calculating E[X|y] in each bin. Fang and Zhu (1996)
    proposed replacing the bins by a kernel smoothing method. The central subspace
    is determined by performing a weighted principal component analysis on the
    kernel estimate of E[X|y]. Note that for the best results a method such
    as cross-validation should be used to choose the bandwidth of the kernel.

    Parameters
    ----------
    n_directions : int, str or None (default='auto')
        Number of directions to keep. Corresponds to the dimension of
        the central subpace. If n_directions==None,
        the number of directions equals the number of features.

    kernel : string or callable, default='rbf'
        Kernel mapping used internally. A callable should accept two arguments
        and the keyword arguments passed to his object as kernel_params, and
        should return a floating point number.

    gamma : float, default='auto'
        Gamma parameter for the RBF, laplacian, polynomial, exponential, chi2,
        and sigmoid kernels. If gamma='auto, the default for the RBF kernel
        is an (0.75 * n_samples) ** (-0.2), i.e. an undersmoothed version of
        the optimal kernel if this was a non-parametric regression problem.

    degree : float, default=2
        Degree of the polynomial kernel. Ignored by other kernels.

    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : mapping of string to any, optional
        Additional parameters (keyword arguments) for a kernel function
        passed as a callable object.

    epsilon : float, default=0.1
        Regularization parameter used to avoid division by zero in the
        kernel estimator of E[X|y].

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

    [1] Fang, K., Zhu, L. (1996),
        "Asymptotics of kernel estimate of sliced inverse regression",
        Annals of Statistics, 24, 1053-1068

    """
    def __init__(self,
                 n_directions=None,
                 kernel='rbf',
                 gamma='auto',
                 degree=2,
                 coef0=1,
                 kernel_params=None,
                 epsilon=1e-1,
                 copy=True):
        self.n_directions = n_directions
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.epsilon = epsilon
        self.copy = copy

    def _get_gamma(self, y):
        if isinstance(self.gamma, six.string_types):
            # undersmoothed optimal bandwidth (n^(-1/(1 + 4)))
            n_samples = y.shape[0]
            gamma = (0.75 * n_samples) ** (-0.2)
            return 1 / (gamma * 2)
        return self.gamma

    def _get_kernel(self, y):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self._get_gamma(y),
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(y.reshape(-1, 1), metric=self.kernel,
                                filter_params=True, **params)

    def fit(self, X, y, kernel=None):
        """Fit the model with X and y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target values (class labels in classification, real numbers
            in regression).

        kernel : array-like, shape (n_samples, n_samples)
            Precomputed kernel matrix.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if sparse.issparse(X):
            raise TypeError("OuterProductGradient does not support "
                            "sparse input.")

        X, y = check_X_y(X, y, dtype=[np.float64, np.float32],
                         y_numeric=True, copy=self.copy)

        n_samples, n_features = X.shape

        self.n_directions_ = (n_features if self.n_directions is None else
                              self.n_directions)

        # validate y
        if is_multioutput(y):
            raise TypeError("The target `y` cannot be multi-output.")

        # Center and Whiten feature matrix using a QR decomposition
        # (this is the approach used in the dr package)
        self.mean_ = np.mean(X, axis=0)
        if self.copy:
            X = X - self.mean_
        else:
            X -= self.mean_
        Q, R = linalg.qr(X, mode='economic')
        Z = np.sqrt(n_samples) * Q

        # standardize target
        y = (y - np.mean(y)) / np.std(y)

        # kernel function
        K = self._get_kernel(y) if kernel is None else kernel

        # kernel version of var(E[X|y])
        K_mean = np.mean(K, axis=1)
        scale = np.mean(K_mean)
        weights = 1 / np.maximum(K_mean, self.epsilon * scale) ** 2
        EZy = np.dot(K, Z) * weights.reshape(-1, 1)
        M = np.dot(EZy.T, EZy)

        # solve eigen-value problem, i.e. outer product of gradient estimates
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

        directions = normalize(
            directions[:, :self.n_directions_], norm='l2', axis=0)
        self.directions_ = directions.T
        self.eigenvalues_ = evals[:self.n_directions_]
        self.eigenvectors_ = evecs[:, :self.n_directions_].T

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
