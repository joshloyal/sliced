from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from sliced import IterativeHessianTransformation
from sliced import datasets


def test_quadratic_response():
    X, y = datasets.make_quadratic(random_state=123)

    iht = IterativeHessianTransformation(n_directions=4).fit(X, y)
    X_iht = iht.transform(X)
    assert X_iht.shape == (X.shape[0], 4)

    true_beta = (1 / np.sqrt(2)) * np.hstack((np.ones(2), np.zeros(8)))
    angle = np.dot(true_beta, iht.directions_[0, :])
    np.testing.assert_allclose(np.abs(angle), 1, rtol=1e-1)


def test_quadratic_residual():
    X, y = datasets.make_quadratic(random_state=123)

    iht = IterativeHessianTransformation(n_directions=4,
                                         target_type='residual')
    iht.fit(X, y)
    X_iht = iht.transform(X)
    assert X_iht.shape == (X.shape[0], 4)

    true_beta = (1 / np.sqrt(2)) * np.hstack((np.ones(2), np.zeros(8)))
    angle = np.dot(true_beta, iht.directions_[0, :])
    np.testing.assert_allclose(np.abs(angle), 1, rtol=1e-1)
