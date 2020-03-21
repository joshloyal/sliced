import numpy as np

from sklearn.base import clone, BaseEstimator, MetaEstimatorMixin
from sklearn.base import TransformerMixin
from sklearn.utils import resample, check_random_state, check_array


__all__ = ['BICDimensionSelector']


class BICDimensionSelector(MetaEstimatorMixin, BaseEstimator, TransformerMixin):
    def __init__(self, estimator, criterion='log'):
        self.estimator = estimator
        self.criterion = criterion

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # fit sufficient dimension reduction method
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)

        evals = self.estimator_.eigenvalues_
        self.criterion_ = np.zeros(n_features + 1)
        if self.criterion == 'sum':
            penalty = 2 * evals[0] * ((n_samples)**(-0.5)) * np.log(n_samples)
            for k in range(n_features + 1):
                if k == 0:
                    self.criterion_[k] = 0
                else:
                    self.criterion_[k] = np.sum(evals[:k])
                    self.criterion_[k] -= penalty * k
        elif self.criterion == 'log':
            penalty = (evals[0] / 3.) * (np.log(n_samples) + np.sqrt(n_samples))
            penalty /= (2. * n_samples)
            for k in range(n_features + 1):
                if k != n_features:
                    self.criterion_[k] = np.sum(np.log(evals[k:] + 1) - evals[k:])
                self.criterion_[k] -= penalty * k * (2 * n_features - k + 1)
        else:
            raise ValueError('Unrecognized criterion {}'.format(self.criterion))

        self.n_directions_ = np.argmax(self.criterion_)

        return self

    def transform(self, X):
        return self.estimator_.transform(X)
