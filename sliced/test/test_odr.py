from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from sliced import OuterProductGradient
from sliced import datasets


def test_cubic():
    X, y = datasets.make_cubic(random_state=123)

    odr = OuterProductGradient(n_directions=2).fit(X, y)
    print(odr.directions_)

    odr.transform(X)
