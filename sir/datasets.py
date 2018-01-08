import numpy as np

from sklearn.utils import check_random_state


def make_cubic(n_samples=500, n_features=10, n_informative=2,
               random_state=None):
    rng = check_random_state(random_state)

    if n_features < n_informative:
        raise ValueError("`n_features` must be >= `n_informative`. "
                         "Got n_features={0} and n_informative={1}".format(
                            n_features, n_informative))

    # normally distributed features
    X = rng.randn(n_samples, n_features)

    # beta is a linear combination of informative features
    beta = np.hstack((
        np.ones(n_informative), np.zeros(n_features - n_informative)))

    # cubic in subspace
    y = 0.125 * np.dot(X, beta) ** 3
    y += 0.5 * rng.randn(n_samples)

    return X, y


def make_quadratic(n_samples=500, n_features=10, n_informative=2,
               random_state=None):
    rng = check_random_state(random_state)

    if n_features < n_informative:
        raise ValueError("`n_features` must be >= `n_informative`. "
                         "Got n_features={0} and n_informative={1}".format(
                            n_features, n_informative))

    # normally distributed features
    X = rng.randn(n_samples, n_features)

    # beta is a linear combination of informative features
    beta = np.hstack((
        np.ones(n_informative), np.zeros(n_features - n_informative)))

    # cubic in subspace
    y = 0.125 * np.dot(X, beta) ** 2
    y += 0.25 * rng.randn(n_samples)

    return X, y

