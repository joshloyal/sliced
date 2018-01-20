from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from sliced.base import whiten_X


rng = np.random.RandomState(123)

sigma = np.array([[1.0, 0.5],
                  [0.5, 1.0]])
mean = np.array([-2, 5])

X = rng.multivariate_normal(mean, sigma, 100)


def test_whiten_cholesky():
    Z, sigma_inv = whiten_X(X, method='cholesky')

    Z_mean = np.mean(Z, axis=0)
    np.testing.assert_allclose(Z_mean, np.zeros(2), atol=1e-7)

    Z_cov = np.dot(Z.T, Z) / (Z.shape[0] - 1)
    np.testing.assert_allclose(Z_cov, np.eye(2), atol=1e-7)


def test_whiten_qr():
    Z, sigma_inv = whiten_X(X, method='qr')

    Z_mean = np.mean(Z, axis=0)
    np.testing.assert_allclose(Z_mean, np.zeros(2), atol=1e-7)

    # QR is much less stable than the other two methods
    Z_cov = np.dot(Z.T, Z) / (Z.shape[0] - 1)
    np.testing.assert_allclose(Z_cov, np.eye(2), rtol=1e-2, atol=1e-2)


def test_whiten_spectral():
    Z, sigma_inv = whiten_X(X, method='spectral')

    Z_mean = np.mean(Z, axis=0)
    np.testing.assert_allclose(Z_mean, np.zeros(2), atol=1e-7)

    Z_cov = np.dot(Z.T, Z) / (Z.shape[0] - 1)
    np.testing.assert_allclose(Z_cov, np.eye(2), atol=1e-7)
