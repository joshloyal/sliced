import numpy as np
import scipy.linalg as linalg


def whiten_X(X, method='cholesky'):
    n_samples = X.shape[0]

    # center data
    X -= np.mean(X, axis=0)

    sigma = np.dot(X.T, X) / (n_samples - 1)

    if method == 'cholesky':
        L = linalg.cholesky(sigma)
        sigma_inv = linalg.solve_triangular(L, np.eye(L.shape[0]))
        Z = np.dot(X, sigma_inv)
    elif method == 'qr':
        Q, R = linalg.qr(np.dot(np.eye(n_samples), X), mode='economic')
        sigma_inv = linalg.solve_triangular(R, np.eye(R.shape[0]))
        Z = np.sqrt(n_samples) * Q
    elif method == 'spectral':
        sigma_inv = linalg.inv(linalg.sqrtm(sigma))
        Z = np.dot(X, sigma_inv)
    else:
        raise ValueError(
            "Unrecoginized whitening method={}. Must be one of "
            "{'cholesky', 'qr', 'spectral'}.".format(method))

    return Z, sigma_inv


def slice_X(X, n_slices):
    n_samples = X.shape[0]

    slices = np.repeat(np.arange(n_slices),
                       np.ceil(n_samples / n_slices))[:n_samples]
    counts = np.bincount(slices)

    return slices, counts
