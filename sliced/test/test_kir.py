from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from sliced import KernelInverseRegression
from sliced import datasets


def test_cubic():
    X, y = datasets.make_cubic(random_state=123)

    kir = KernelInverseRegression(n_directions=4).fit(X, y)
    X_kir = kir.transform(X)
    assert X_kir.shape == (X.shape[0], 4)

    true_beta = (1 / np.sqrt(2)) * np.hstack((np.ones(2), np.zeros(8)))
    angle = np.dot(true_beta, kir.directions_[0, :])
    np.testing.assert_allclose(np.abs(angle), 1, rtol=1e-1)
