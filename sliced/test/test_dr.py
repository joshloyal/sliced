from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import pytest

from scipy import sparse
from scipy import linalg

from sliced import DirectionalRegression
from sliced import datasets


def test_cubic():
    X, y = datasets.make_cubic(random_state=123)

    dr = DirectionalRegression(n_slices=3).fit(X, y)

    true_beta = (1 / np.sqrt(2)) * np.hstack((np.ones(2), np.zeros(8)))
    angle = np.dot(true_beta, dr.directions_[0, :])
    np.testing.assert_allclose(np.abs(angle), 1, rtol=1e-1)
