import numpy as np

from sklearn.base import clone, BaseEstimator, MetaEstimatorMixin
from sklearn.base import TransformerMixin
from sklearn.utils import resample, check_random_state, check_array


__all__ = ['LadleDimensionSelector']


class LadleDimensionSelector(MetaEstimatorMixin, BaseEstimator, TransformerMixin):
    def __init__(self, estimator, n_boot=200, random_state=None):
        self.estimator = estimator
        self.n_boot = n_boot
        self.random_state = random_state

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # calculate k_max
        self.k_max_ = (n_features - 2 if n_features <= 10 else
            int(n_features / np.log(n_features)))

        # fit sufficient dimension reduction method
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)

        # matrix with eigenvectors as columns
        B = self.estimator_.eigenvectors_.T
        evals = self.estimator_.eigenvalues_

        # bootstrap eigenvector variation statistic
        random_state = check_random_state(self.random_state)
        D = np.zeros((self.n_boot, self.k_max_))
        for b in range(self.n_boot):
            X_boot, y_boot = resample(X, y, random_state=random_state)
            B_boot = clone(self.estimator).fit(X_boot, y_boot).eigenvectors_.T
            for k in range(self.k_max_):
                if k > 0:
                    D[b, k] = 1 - np.abs(
                        np.linalg.det(
                            np.dot(B[:, :(k+1)].T, B_boot[:, :(k+1)])))
                else:
                    D[b, k] = 1 - np.abs(np.dot(B[:, 0].T, B_boot[:, 0]))

        # calculate test statistics
        eigvec_variation = np.r_[0, np.mean(D, axis=0)]
        eigvec_variation /= (1 + np.sum(eigvec_variation))
        eigval_magnitude = (evals[:(self.k_max_ + 1)] /
            (1 + np.sum(evals[:(self.k_max_ + 1)])))

        # record number directions and ladle criterion values for each k
        self.criterion_ = eigvec_variation + eigval_magnitude
        self.n_directions_ = np.argmin(self.criterion_)

        return self

    def transform(self, X):
        return self.estimator_.transform(X)
