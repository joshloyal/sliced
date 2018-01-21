from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from scipy import sparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sliced import SlicedInverseRegression
from sliced import datasets


def test_regression():
    X, y = datasets.make_cubic(random_state=123)

    for n_comp in range(X.shape[1]):
        sir = SlicedInverseRegression(n_components=n_comp)

        # take shape is correct
        X_sir = sir.fit(X, y).transform(X)
        np.testing.assert_equal(X_sir.shape[1], n_comp)

        # should match fit_transform
        X_sir2 = sir.fit_transform(X, y)
        np.testing.assert_allclose(X_sir, X_sir2)

        # call transform again and check if things are okay
        X_sir = sir.transform(X)
        X_sir2 = sir.fit_transform(X, y)
        np.testing.assert_allclose(X_sir, X_sir2)

        # there is one true angle it should fine
        if n_comp > 0:
            true_beta = (1 / np.sqrt(2)) * np.hstack((np.ones(2), np.zeros(8)))
            angle = np.dot(true_beta, sir.components_[0, :])
            np.testing.assert_allclose(np.abs(angle), 1, rtol=1e-1)


def test_classification():
    """SIR is LDA for classification so lets test some predictions."""
    # Data is just 6 separable points in the plane
    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]],
                 dtype=np.float64)
    y = np.array([1, 1, 1, 0, 0, 0])

    sir = SlicedInverseRegression(n_components=1, n_slices=2).fit(X, y)

    y_pred = sir.transform(X) > 0.5
    np.testing.assert_equal(y, y_pred.ravel())


def test_n_components_none():
    X, y = datasets.make_cubic(random_state=123)
    sir = SlicedInverseRegression(n_components=None).fit(X, y)
    np.testing.assert_equal(sir.n_components_, X.shape[1])


def test_single_y_value():
    rng = np.random.RandomState(123)

    X = rng.randn(100, 4)
    y = np.ones(100)

    with pytest.raises(ValueError):
        SlicedInverseRegression().fit(X, y)


def test_sparse_not_supported():
    X, y = datasets.make_cubic(random_state=123)
    X = sparse.csr_matrix(X)
    with pytest.raises(TypeError):
        SlicedInverseRegression().fit(X, y)


def test_n_components_auto_heuristic():
    X, y = datasets.make_exponential(random_state=123)
    sir = SlicedInverseRegression(n_components='auto').fit(X, y)
    assert sir.n_components_ == 2

    X_sir = sir.transform(X)
    assert X_sir.shape == (500, 2)
