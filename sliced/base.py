from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.linalg as linalg


def is_multioutput(y):
    """Whether the target y is multi-output (or multi-index)"""
    return hasattr(y, "shape") and y.ndim == 2 and y.shape[1] > 1


def whiten_X(X, center=True, method='cholesky', copy=False):
    """Whiten a data matrix using either the choleksy, QR or
    spectral method."""
    n_samples = X.shape[0]

    # center data
    if center:
        means = np.mean(X, axis=0)
        if copy:
            X = X - np.mean(X, axis=0)
        else:
            X -= means

    if method == 'cholesky':
        sigma = np.dot(X.T, X) / (n_samples - 1)
        L = linalg.cholesky(sigma)
        sigma_inv = linalg.solve_triangular(L, np.eye(L.shape[0]))
        Z = np.dot(X, sigma_inv)
    elif method == 'qr':
        Q, R = linalg.qr(np.dot(np.eye(n_samples), X), mode='economic')
        sigma_inv = linalg.solve_triangular(R, np.eye(R.shape[0]))
        Z = np.sqrt(n_samples) * Q
    elif method == 'spectral':
        sigma = np.dot(X.T, X) / (n_samples - 1)
        sigma_inv = linalg.inv(linalg.sqrtm(sigma))
        Z = np.dot(X, sigma_inv)
    else:
        raise ValueError(
            "Unrecoginized whitening method={}. Must be one of "
            "{'cholesky', 'qr', 'spectral'}.".format(method))

    return Z, sigma_inv


def slice_X(X, n_slices):
    """Create equal slices of a matrix."""
    n_samples = X.shape[0]

    slices = np.repeat(np.arange(n_slices),
                       np.ceil(n_samples / n_slices))[:n_samples]
    counts = np.bincount(slices)

    return slices, counts
