import numpy as np
import scipy.linalg as linalg


def whiten_X(X, return_inverse=True):
    n_samples = X.shape[0]

    # center data
    X -= np.mean(X, axis=0)

    sigma = np.dot(X.T, X) / (n_samples - 1)
    L = linalg.cholesky(sigma)
    L_inv = linalg.solve_triangular(L, np.eye(L.shape[0]))
    Z = np.dot(X, L_inv)

    return Z, L_inv


def slice_X(X, n_slices):
    n_samples = X.shape[0]

    slices = np.repeat(np.arange(n_slices),
                       np.ceil(n_samples / n_slices))[:n_samples]
    counts = np.bincount(slices)

    return slices, counts
