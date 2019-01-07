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


__all__ = ['OuterProductGradient']


class OuterProductGradient(BaseEstimator, TransformerMixin):
    def __init__(self,
                 n_directions='auto',
                 kernel='rbf',
                 gamma='auto',
                 degree=2,
                 coef0=1,
                 kernel_params=None,
                 copy=True):
        self.n_directions = n_directions
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.copy = copy

    def _get_gamma(self, X):
        n_samples, n_features = X.shape
        if isinstance(self.gamma, six.string_types):
            # bandwidth proposed in Xia (2007)
            gamma = 2.34 * n_samples ** (-1 /
                                          (np.maximum(n_features, 3) + 6))
            return 1 / (gamma * 2)
        return self.gamma

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self._get_gamma(X),
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

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
            raise TypeError("OuterProductGradient does not support "
                            "sparse input.")

        X, y = check_X_y(X, y, dtype=[np.float64, np.float32],
                         y_numeric=True, copy=self.copy)

        n_samples, n_features = X.shape

        if isinstance(self.n_directions, six.string_types):
            self.n_directions_ = X.shape[1]
        else:
            self.n_directions_ = self.n_directions

        # validate y
        if is_multioutput(y):
            raise TypeError("The target `y` cannot be multi-output.")

        # Center and scale feature matrix
        scaler = StandardScaler(copy=self.copy)
        X = scaler.fit_transform(X)

        # pre-compute the kernel weights
        K = self._get_kernel(X)

        # calculate the gradient estimate through a local wls
        gradient = np.zeros((n_features, n_samples))
        for i in range(n_samples):
            linear_model = LinearRegression(fit_intercept=True)
            linear_model.fit(X - X[i, :], y, sample_weight=K[i, :])
            gradient[:, i] = linear_model.coef_

        # solve eigen-value problem, i.e. outer product of gradient estimates
        M = np.dot(gradient, gradient.T)
        evals, evecs = linalg.eigh(M)
        evecs = evecs[:, ::-1]
        evals = evals[::-1]

        # convert back to un-scaled basis
        directions = evecs / np.sqrt(scaler.var_)
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
        return np.dot(X, self.directions_.T)
