from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from sklearn.utils import check_random_state


__all__ = ['make_cubic', 'make_quadratic', 'make_polynomial',
           'make_exponential']


def make_cubic(n_samples=500, n_features=10, n_informative=2,
               random_state=None):
    """Generates a dataset with a cubic response curve.

    Inputs X are independent normally distributed features. The output y
    is created according to the formula::

        beta = np.hstack((
            np.ones(n_informative), np.zeros(n_features - n_informative)))
        h = np.dot(X, beta)
        y(h) = 0.125 * h ** 3 + 0.5 * N(0, 1).

    Out of the `n_features` features,  only `n_informative` are actually
    used to compute `y`. The remaining features are independent of `y`.
    As such the central subspace is one dimensional and consists of the
    `h` axis.

    Parameters
    ----------
    n_samples : int, optimal (default=500)
        The number of samples.

    n_features : int, optional (default=10)
        The number of features. Should be at least equal to `n_informative`.

    n_informative : int, optional (default=2)
        The number of informative features used to construct h. Should be
        at least 1.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The input samples.

    y : array of shape [n_samples]
        The output values.
    """
    rng = check_random_state(random_state)

    if n_informative < 1:
        raise ValueError("`n_informative` must be >= 1. "
                         "Got n_informative={0}".format(n_informative))

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
    """Generates a dataset with a quadratic response curve.

    Inputs X are independent normally distributed features. The output y
    is created according to the formula::

        beta = np.hstack((
            np.ones(n_informative), np.zeros(n_features - n_informative)))
        h = np.dot(X, beta)
        y(h) = 0.125 * h ** 2 + 0.5 * N(0, 1).

    Out of the `n_features` features,  only `n_informative` are actually
    used to compute `y`. The remaining features are independent of `y`.
    As such the central subspace is one dimensional and consists of the
    `h` axis.

    Parameters
    ----------
    n_samples : int, optimal (default=500)
        The number of samples.

    n_features : int, optional (default=10)
        The number of features. Should be at least equal to `n_informative`.

    n_informative : int, optional (default=2)
        The number of informative features used to construct h. Should be
        at least 1.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The input samples.

    y : array of shape [n_samples]
        The output values.
    """
    rng = check_random_state(random_state)

    if n_informative < 1:
        raise ValueError("`n_informative` must be >= 1. "
                         "Got n_informative={0}".format(n_informative))

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


def make_polynomial(n_samples=500, n_features=10, random_state=None):
    """Generates a dataset with a polynomial response curve that combines
    a quadratic and cubic response. There are two primary dimensions instead
    of one.

    Inputs X are independent normally distributed features. The output y
    is created according to the formula::

        beta1 = np.hstack((
            [1, 1, 1], np.zeros(n_features - 3)))
        beta2 = np.hstack((
            [1, -1, -1], np.zeros(n_features - 3)))

        u = np.dot(X, beta1)
        v = np.dot(X, beta2)
        y(u, v) = u + u ** 3 + v ** 2 + N(0, 1)

    Out of the `n_features` features,  only 3 are actually
    used to compute `y`. The remaining features are independent of `y`.
    As such the central subspace is two dimensional and consists of the
    `u` and `v` axes.

    Parameters
    ----------
    n_samples : int, optimal (default=500)
        The number of samples.

    n_features : int, optional (default=10)
        The number of features. Should be at least equal to 3, the number
        of informative features.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The input samples.

    y : array of shape [n_samples]
        The output values.
    """
    rng = check_random_state(random_state)

    if n_features < 3:
        raise ValueError("`n_features` must be >= 3. "
                         "Got n_features={0}".format(n_features))

    # normally distributed features
    X = rng.randn(n_samples, n_features)

    # beta is a linear combination of informative features
    beta1 = np.hstack((
        [1, 1, 1], np.zeros(n_features - 3)))
    beta2 = np.hstack((
        [1, -1, -1], np.zeros(n_features - 3)))

    u = np.dot(X, beta1)
    v = np.dot(X, beta2)
    y = 0.7 * (u + u ** 3) + v ** 2
    y += 0.5 * rng.randn(n_samples)

    return X, y


def make_exponential(n_samples=500, n_features=20, random_state=None):
    """Generates a dataset with a exponential response curve.

    Inputs X are independent normally distributed features. The output y
    is created according to the formula::

        beta1 = np.hstack((
            np.ones(7), np.zeros(n_features - 7)))
        beta2 = np.hstack((
            np.zeros(7), np.ones(4), np.zeros(n_features - 11)))
        u = np.dot(X, beta1)
        v = np.dot(X, beta2)
        y(u, v) = u * np.exp(v) + N(0, 1)

    Parameters
    ----------
    n_samples : int, optimal (default=500)
        The number of samples.

    n_features : int, optional (default=10)
        The number of features. Should be at least equal to `n_informative`.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The input samples.

    y : array of shape [n_samples]
        The output values.
    """
    rng = check_random_state(random_state)

    if n_features < 11:
        raise ValueError("`n_features` must be >= 11. "
                         "Got n_features={0}".format(n_features))

    X = rng.randn(n_samples, 20)

    beta1 = np.hstack((
        np.ones(7), np.zeros(n_features - 7)))
    beta2 = np.hstack((
        np.zeros(7), np.ones(4), np.zeros(n_features - 11)))

    u = np.dot(X, beta1)
    v = np.dot(X, beta2)
    y = u * np.exp(v)
    y += rng.randn(n_samples)

    return X, y
