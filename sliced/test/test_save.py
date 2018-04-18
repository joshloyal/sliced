from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import pytest

from scipy import sparse
from scipy import linalg
from sklearn.datasets import load_digits

from sliced import SlicedAverageVarianceEstimation
from sliced import datasets


def test_cubic():
    X, y = datasets.make_cubic(random_state=123)

    save = SlicedAverageVarianceEstimation().fit(X, y)

    true_beta = (1 / np.sqrt(2)) * np.hstack((np.ones(2), np.zeros(8)))
    angle = np.dot(true_beta, save.directions_[0, :])
    np.testing.assert_allclose(np.abs(angle), 1, rtol=1e-1)


def test_regression():
    """NOTE: subsequent calls may flip the direction of eigenvectors
        (mulitply by -1), so we can only compare absolute values.

        This was not a problem for svds.. investigate if we can get
        deterministic behavior back.
    """
    X, y = datasets.make_quadratic(random_state=123)

    for n_dir in range(1, X.shape[1]):
        save = SlicedAverageVarianceEstimation(n_directions=n_dir)

        # take shape is correct
        X_save = save.fit(X, y).transform(X)
        np.testing.assert_equal(X_save.shape[1], n_dir)

        # should match fit_transform
        X_save2 = save.fit_transform(X, y)
        np.testing.assert_allclose(np.abs(X_save), np.abs(X_save2))

        # call transform again and check if things are okay
        X_save = save.transform(X)
        X_save2 = save.fit_transform(X, y)
        np.testing.assert_allclose(np.abs(X_save), np.abs(X_save2))

        # there is one true angle it should fine
        true_beta = (1 / np.sqrt(2)) * np.hstack((np.ones(2), np.zeros(8)))
        angle = np.dot(true_beta, save.directions_[0, :])
        np.testing.assert_allclose(np.abs(angle), 1, rtol=1e-1)


def test_n_directions_none():
    X, y = datasets.make_cubic(random_state=123)
    sir = SlicedAverageVarianceEstimation(n_directions=None).fit(X, y)
    np.testing.assert_equal(sir.n_directions_, X.shape[1])


def test_n_slices_too_big():
    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]],
                 dtype=np.float64)
    y = np.array([1, 1, 1, 0, 0, 0])

    save = SlicedAverageVarianceEstimation(n_directions=1, n_slices=10)
    save.fit(X, y)

    assert save.n_slices_ == 2


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


def test_n_directions_auto_heuristic():
    X, y = datasets.make_exponential(random_state=123)
    save = SlicedAverageVarianceEstimation(n_directions='auto').fit(X, y)
    assert save.n_directions_ == 2

    X_save = save.transform(X)
    assert X_save.shape == (500, 2)


def test_zero_variance_features():
    """Raise an informative error message when features of zero variance."""
    X, y = load_digits(return_X_y=True)

    with pytest.raises(linalg.LinAlgError):
        save = SlicedAverageVarianceEstimation(n_directions='auto').fit(X, y)


@pytest.mark.skipif(sys.platform == 'win32',
                    reason=("Lapack's eigh is not deterministic across ",
                            "platforms. The sign of some eigenvectors is ",
                            "flipped on win32."))
def test_matches_swiss_banknote():
    """Test that the results match the R dr package on a few common datasets.
    """
    X, y = datasets.load_banknote()
    save = SlicedAverageVarianceEstimation(n_directions=4).fit(X, y)

    np.testing.assert_allclose(
        save.eigenvalues_,
        np.array([0.87239404, 0.42288351, 0.12792117, 0.03771284])
    )

    expected_directions = np.array(
        [[0.03082069, 0.20309393, -0.25314643, -0.58931337, -0.56801632,
          0.47306135],
         [-0.2841728, -0.05472057, -0.15731808, 0.50606843, 0.33404888,
          0.72374622],
         [0.09905744, -0.88896348, 0.42252244, -0.00162151, -0.09222179,
          -0.11357311],
         [0.75251819, -0.26448055, 0.59669025, 0.03982343, -0.018666,
          0.07611073]],
    )
    np.testing.assert_allclose(
        save.directions_, expected_directions, atol=1e-8)
