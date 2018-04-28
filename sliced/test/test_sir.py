from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import pytest

from scipy import sparse
from scipy import linalg
from sklearn.datasets import load_digits
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sliced import SlicedInverseRegression
from sliced import datasets


def test_regression():
    """NOTE: subsequent calls may flip the direction of eigenvectors
        (mulitply by -1), so we can only compare absolute values.

        This was not a problem for svds.. investigate if we can get
        deterministic behavior back.
    """
    X, y = datasets.make_cubic(random_state=123)

    for n_dir in range(1, X.shape[1]):
        sir = SlicedInverseRegression(n_directions=n_dir)

        # take shape is correct
        X_sir = sir.fit(X, y).transform(X)
        np.testing.assert_equal(X_sir.shape[1], n_dir)

        # should match fit_transform
        X_sir2 = sir.fit_transform(X, y)
        np.testing.assert_allclose(np.abs(X_sir), np.abs(X_sir2))

        # call transform again and check if things are okay
        X_sir = sir.transform(X)
        X_sir2 = sir.fit_transform(X, y)
        np.testing.assert_allclose(np.abs(X_sir), np.abs(X_sir2))

        # there is one true angle it should fine
        true_beta = (1 / np.sqrt(2)) * np.hstack((np.ones(2), np.zeros(8)))
        angle = np.dot(true_beta, sir.directions_[0, :])
        np.testing.assert_allclose(np.abs(angle), 1, rtol=1e-1)


def test_classification():
    """SIR is LDA for classification so lets test some predictions."""
    # Data is just 6 separable points in the plane
    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]],
                 dtype=np.float64)
    y = np.array([1, 1, 1, 0, 0, 0])

    sir = SlicedInverseRegression(n_directions=1, n_slices=2).fit(X, y)
    lda = LinearDiscriminantAnalysis(solver='eigen').fit(X, y)

    y_pred = sir.transform(X) > 0
    np.testing.assert_equal(y, y_pred.ravel())
    np.testing.assert_equal(lda.predict(X), y_pred.ravel())


def test_n_slices_too_big():
    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]],
                 dtype=np.float64)
    y = np.array([1, 1, 1, 0, 0, 0])

    sir = SlicedInverseRegression(n_directions=1, n_slices=10).fit(X, y)

    assert sir.n_slices_ == 2


def test_n_directions_none():
    X, y = datasets.make_cubic(random_state=123)
    sir = SlicedInverseRegression(n_directions=None).fit(X, y)
    np.testing.assert_equal(sir.n_directions_, X.shape[1])


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


def test_n_directions_auto_heuristic():
    X, y = datasets.make_exponential(random_state=123)
    sir = SlicedInverseRegression(n_directions='auto').fit(X, y)
    assert sir.n_directions_ == 2

    X_sir = sir.transform(X)
    assert X_sir.shape == (500, 2)


def test_zero_variance_features():
    """Raise an informative error message when features of zero variance."""
    X, y = load_digits(return_X_y=True)

    with pytest.raises(linalg.LinAlgError):
        sir = SlicedInverseRegression(n_directions='auto').fit(X, y)


@pytest.mark.skipif(sys.platform == 'win32',
                    reason=("Lapack's eigh is not deterministic across ",
                            "platforms. The sign of some eigenvectors is ",
                            "flipped on win32."))
def test_matches_athletes():
    """Test that the resutls match the R dr package on a ais dataset.
    """
    X, y = datasets.load_athletes()
    sir = SlicedInverseRegression(n_directions=4, n_slices=11).fit(X, y)

    np.testing.assert_allclose(
        sir.eigenvalues_,
        np.array([0.957661631, 0.245041613, 0.107075941, 0.090413047])
    )

    expected_directions = np.array(
        [[1.50963358e-01,  -9.16480522e-01,  -1.31538894e-01,  -9.33588596e-02,
          4.46783829e-03,  -1.88973540e-01,   2.74758965e-01,  -5.63123794e-03],
         [-5.01785457e-02,  -1.94229862e-01,   6.85475076e-01,  -4.33408964e-02,
          1.83380846e-04,   3.47565293e-01,  -6.05830142e-01,   1.30588502e-02],
         [1.08983356e-01,  -2.01236965e-01,   7.19975455e-01,   4.64453982e-01,
          4.49759016e-02,   2.94969081e-01,  -3.41966152e-01,  -8.70270913e-02],
         [-2.21020634e-03,  -8.97220257e-02,  -6.63097774e-01,   2.90838658e-01,
          7.19045566e-02,   3.70563626e-02,   6.78877114e-01,   1.55472144e-02]]
    )

    np.testing.assert_allclose(
        sir.directions_, expected_directions)
