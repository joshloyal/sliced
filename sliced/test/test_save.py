from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pytest

from scipy import sparse

from sliced import SlicedAverageVarianceEstimation
from sliced import datasets


def test_cubic():
    X, y = datasets.make_cubic(random_state=123)

    save = SlicedAverageVarianceEstimation().fit(X, y)

    true_beta = (1 / np.sqrt(2)) * np.hstack((np.ones(2), np.zeros(8)))
    angle = np.dot(true_beta, save.components_[0, :])
    np.testing.assert_allclose(np.abs(angle), 1, rtol=1e-1)


def test_regression():
    X, y = datasets.make_quadratic(random_state=123)

    for n_comp in range(X.shape[1]):
        save = SlicedAverageVarianceEstimation(n_components=n_comp)

        # take shape is correct
        X_save = save.fit(X, y).transform(X)
        np.testing.assert_equal(X_save.shape[1], n_comp)

        # should match fit_transform
        X_save2 = save.fit_transform(X, y)
        np.testing.assert_allclose(X_save, X_save2)

        # call transform again and check if things are okay
        X_save = save.transform(X)
        X_save2 = save.fit_transform(X, y)
        np.testing.assert_allclose(X_save, X_save2)

        # there is one true angle it should fine
        if n_comp > 0:
            true_beta = (1 / np.sqrt(2)) * np.hstack((np.ones(2), np.zeros(8)))
            angle = np.dot(true_beta, save.components_[0, :])
            np.testing.assert_allclose(np.abs(angle), 1, rtol=1e-2)


def test_classification():
    """Like SIR we can use SAVE for classification (like LDA)"""
    # Data is just 6 separable points in the plane
    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]],
                 dtype=np.float64)
    y = np.array([1, 1, 1, 0, 0, 0])

    sir = SlicedAverageVarianceEstimation(n_components=1, n_slices=2).fit(X, y)
    y_pred = sir.transform(X) > 0.5
    np.testing.assert_equal(y, y_pred.ravel())


def test_n_components_none():
    X, y = datasets.make_cubic(random_state=123)
    sir = SlicedAverageVarianceEstimation(n_components=None).fit(X, y)
    np.testing.assert_equal(sir.n_components_, X.shape[1])


def test_single_y_value():
    rng = np.random.RandomState(123)

    X = rng.randn(100, 4)
    y = np.ones(100)

    with pytest.raises(ValueError):
        SlicedAverageVarianceEstimation().fit(X, y)


def test_sparse_not_supported():
    X, y = datasets.make_cubic(random_state=123)
    X = sparse.csr_matrix(X)

    with pytest.raises(TypeError):
        SlicedAverageVarianceEstimation().fit(X, y)
