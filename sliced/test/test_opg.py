from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from sliced import OuterProductGradients
from sliced import datasets


def test_cubic():
    X, y = datasets.make_cubic(random_state=123)

    opg = OuterProductGradients(n_directions=4).fit(X, y)
    X_opg = opg.transform(X)
    assert X_opg.shape == (X.shape[0], 4)

    true_beta = (1 / np.sqrt(2)) * np.hstack((np.ones(2), np.zeros(8)))
    angle = np.dot(true_beta, opg.directions_[0, :])
    np.testing.assert_allclose(np.abs(angle), 1, rtol=1e-1)
