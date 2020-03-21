from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from sliced import PrincipalHessianDirections
from sliced import datasets


def test_athletes_matches_dr():
    X, y = datasets.load_athletes()

    phd = PrincipalHessianDirections(n_directions=4).fit(X, y)
    X_phd = phd.transform(X)
    assert X_phd.shape == (X.shape[0], 4)

    expected_directions = np.array(
        [[-0.03675, 0.59536, -0.36061, 0.21613, 0.02948, -0.29816, 0.61429, -0.01761],
        [-0.23340, 0.03252, -0.47699, -0.08133, -0.07203, -0.13669, 0.82846, 0.01068],
        [0.001928, -0.238599, -0.014747, 0.959780, 0.065847, -0.123638, 0.044891,
         -0.005824],
        [0.006563, 0.025140, -0.596972, -0.038954, -0.047897, -0.166642, 0.781899,
        -0.001390]]
    )
    np.testing.assert_allclose(
        np.abs(phd.directions_), np.abs(expected_directions), atol=1e-5)

    expected_eigenvalues = np.array([2.8583, -1.4478, 0.9612, -0.5621])
    np.testing.assert_allclose(phd.eigenvalues_, expected_eigenvalues, atol=1e-4)

def test_response_based_phd():
    X, y = datasets.make_quadratic()

    phd = PrincipalHessianDirections(target_type='response', n_directions=1).fit(X, y)
    X_phd = phd.transform(X)
    assert X_phd.shape == (X.shape[0], 1)

    # check fit_transform matches
    X_phd2 = phd.fit_transform(X, y)
    np.testing.assert_allclose(X_phd, X_phd2)
