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


__all__ = ['IterativeHessianTransformation']


class IterativeHessianTransformation(BaseEstimator, TransformerMixin):
    def __init__(self, n_directions=None, copy=True):
        self.n_directions = n_directions
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

        self.n_directions_ = (n_features if self.n_directions is None else
                              self.n_directions)

        # validate y
        if is_multioutput(y):
            raise TypeError("The target `y` cannot be multi-output.")

        # Center and Whiten feature matrix using a QR decomposition
        # (this is the approach used in the dr package)
        if self.copy:
            X = X - np.mean(X, axis=0)
            #y = y - np.mean(y)
        else:
            X -= np.mean(X, axis=0)
            #y -= np.mean(y)
        Q, R = linalg.qr(X, mode='economic')
        Z = np.sqrt(n_samples) * Q

        lm = LinearRegression(fit_intercept=True).fit(Z, y)
        #Szy = np.zeros(n_features)
        #for p in range(n_features):
        #    Szy[p] = np.cov(Z[:, p], y)[0, 1]
        Szy = lm.coef_

        Zy = Z * (y - np.mean(y) - lm.predict(Z)).reshape(-1, 1)
        #Zy = Z * y.reshape(-1, 1)
        Szzy = np.dot(Zy.T, Z) / n_samples
        #Szzy = np.dot(Z.T, Z) / n_samples
        M = np.zeros((n_features, n_features))
        for p in range(1, n_features):
            M[:, p] = np.dot(np.linalg.matrix_power(Szzy, p), Szy)

        evals, evecs = linalg.eigh(np.dot(M, M.T))
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
