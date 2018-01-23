from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import numpy as np
import scipy.linalg as linalg

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import empirical_covariance, shrunk_covariance
from sklearn.covariance import ledoit_wolf
from sklearn.covariance import GraphLassoCV


def _empirical_covariance(X, assume_centered):
    n_samples = X.shape[0]

    sigma = empirical_covariance(X, assume_centered=assume_centered)
    return (n_samples / (n_samples - 1)) * sigma


def covariance_matrix(X, shrinkage='auto'):
    shrinkage = "empirical" if shrinkage is None else shrinkage
    if isinstance(shrinkage, six.string_types):
        if shrinkage == 'auto':
            sigma, _ = ledoit_wolf(X, assume_centered=True)
        elif shrinkage == 'empirical':
            sigma = _empirical_covariance(X, assume_centered=True)
        else:
            raise ValueError('unknown shrinkage parameter')
    elif isinstance(shrinkage, float) or isinstance(shrinkage, int):
        if shrinkage < 0 or shrinkage > 1:
            raise
        sigma = shrunk_covariance(
            _empirical_covariance(X, assume_centered=True), shrinkage)
    else:
        raise TypeError('shrinkage must be of string or int type')

    return sigma


def sigma_inverse(X, shrinkage='auto', inverse_sqrt=True):
    if shrinkage == 'graph_lasso':
        X = X / X.std(axis=0)
        lasso = GraphLassoCV(assume_centered=True).fit(X)
        if inverse_sqrt:
            return linalg.sqrtm(lasso.precision_)
        return lasso.precision_
    else:
        sigma = covariance_matrix(X, shrinkage=shrinkage)

        if inverse_sqrt:
            sigma = linalg.sqrtm(sigma)

        return linalg.pinvh(sigma)


def whiten_X(X, assume_centered=False, method='cholesky', shrinkage='auto',
             copy=False):
    """Whiten a data matrix using either the choleksy, QR or
    spectral method."""
    n_samples = X.shape[0]

    # center data
    if not assume_centered:
        means = np.mean(X, axis=0)
        if copy:
            X = X - np.mean(X, axis=0)
        else:
            X -= means

    if method == 'cholesky':
        sigma = covariance_matrix(X, shrinkage=shrinkage)
        L = linalg.cholesky(sigma)
        sigma_inv = linalg.solve_triangular(L, np.eye(L.shape[0]))
        Z = np.dot(X, sigma_inv)
    elif method == 'qr':
        Q, R = linalg.qr(np.dot(np.eye(n_samples), X), mode='economic')
        sigma_inv = linalg.solve_triangular(R, np.eye(R.shape[0]))
        Z = np.sqrt(n_samples) * Q
    elif method == 'mahalanobis':
        sigma_inv = sigma_inverse(X, shrinkage=shrinkage, inverse_sqrt=True)
        Z = np.dot(X, sigma_inv)
    elif method == 'pca':
        sigma_inv = sigma_inverse(X, shrinkage=shrinkage, inverse_sqrt=False)
        Z = np.dot(X, sigma_inv)
    else:
        raise ValueError(
            "Unrecoginized whitening method={}. Must be one of "
            "{'cholesky', 'qr', 'mahalanobis', 'pca'}.".format(method))

    return Z, sigma_inv


class Whiten(BaseEstimator, TransformerMixin):
    def __init__(self,
                 assume_centered=False,
                 method='cholesky',
                 shrinkage=None,
                 copy=True):
        self.assume_centered = assume_centered
        self.method = method
        self.shrinkage = shrinkage
        self.copy = copy

    def fit_transform(self, X):
        Z, sigma_inverse = whiten_X(
            X, assume_centered=self.assume_centered, method=self.method,
            shrinkage=self.shrinkage, copy=self.copy)

        self.whitening_matrix_ = sigma_inverse

        return Z
