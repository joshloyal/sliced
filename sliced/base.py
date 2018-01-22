from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import numpy as np


def is_multioutput(y):
    """Whether the target y is multi-output (or multi-index)"""
    return hasattr(y, "shape") and y.ndim == 2 and y.shape[1] > 1


def slice_X(X, n_slices):
    """Create equal slices of a matrix."""
    n_samples = X.shape[0]

    slices = np.repeat(np.arange(n_slices),
                       np.ceil(n_samples / n_slices))[:n_samples]
    counts = np.bincount(slices)

    return slices, counts
