from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pkg_resources import parse_version

import warnings

import numpy as np
import scipy.linalg as linalg


NUMPY_UNIQUE_COUNTS_VERSION = '1.9.0'


def is_multioutput(y):
    """Whether the target y is multi-output (or multi-index)"""
    return hasattr(y, "shape") and y.ndim == 2 and y.shape[1] > 1


def grouped_sum(array, groups):
    """Sums an array by groups. Groups are assumed to be contiguous by row."""
    inv_idx = np.concatenate(([0], np.diff(groups).nonzero()[0] + 1))
    return np.add.reduceat(array, inv_idx)


def unique_counts(arr):
    """Determine the unique values and the number of times they occur in a one
    dimensional array.

    This is a wrapper around numpy's unique function.
    In order to keep the numpy dependency below 1.9 this function falls
    back to the slow version of getting the unique counts array by counting
    the indices of the inverse array.

    Parameters
    ----------
    arr : array_like
        Input array. This array will be flattened if it is not already 1-D.

    Returns
    -------
    unique : ndarray
        The sorted unique values.

    unique_counts : ndarray
        The number of times each of the unique values compes up in the orginal
        array.
    """
    if (parse_version(np.__version__) >=
            parse_version(NUMPY_UNIQUE_COUNTS_VERSION)):
        unique, counts = np.unique(arr, return_counts=True)
    else:
        unique, unique_inverse = np.unique(arr, return_inverse=True)
        counts = np.bincount(unique_inverse)
    return unique, counts


def slice_y(y, n_slices=10):
    """Determine non-overlapping slices based on the target variable, y.

    Parameters
    ----------
    y : array_like, shape (n_samples,)
        The target values (class labels in classification, real numbers
        in regression).

    n_slices : int (default=10)
        The number of slices used when calculating the inverse regression
        curve. Truncated to at most the number of unique values of ``y``.

    Returns
    -------
    slice_indicator : ndarray, shape (n_samples,)
        Index of the slice (from 0 to n_slices) that contains this
        observation.
    slice_counts :  ndarray, shape (n_slices,)
        The number of counts in each slice.
    """
    unique_y_vals, counts = unique_counts(y)
    cumsum_y = np.cumsum(counts)

    # `n_slices` must be less-than or equal to the number of unique values
    # of `y`.
    n_y_values = unique_y_vals.shape[0]
    if n_y_values == 1:
        raise ValueError("The target only has one unique y value. It does "
                         "not make sense to fit SIR or SAVE in this case.")
    elif n_slices >= n_y_values:
        if n_slices > n_y_values:
            warnings.warn(
                "n_slices greater than the number of unique y values. "
                "Setting n_slices equal to {0}.".format(counts.shape[0]))
        # each y value gets its own slice. usually the case for classification
        slice_partition = np.hstack((0, cumsum_y))
    else:
        # attempt to put this many observations in each slice.
        # not always possible since we need to group common y values together
        # NOTE: This should be ceil, but this package is attempting to
        #       replicate the slices used by R's DR package which uses floor.
        n_obs = np.floor(y.shape[0] / n_slices)

        # Loop through the unique y value sums and group
        # slices together with the goal of 2 <= # in slice <= n_obs
        # Find index in y unique where slice begins and ends
        n_samples_seen = 0
        slice_partition = [0]  # index in y of start of a new slice
        while n_samples_seen < y.shape[0] - 2:
            slice_start = np.where(cumsum_y >= n_samples_seen + n_obs)[0]
            if slice_start.shape[0] == 0:  # this means we've reached the end
                slice_start = cumsum_y.shape[0] - 1
            else:
                slice_start = slice_start[0]

            n_samples_seen = cumsum_y[slice_start]
            slice_partition.append(n_samples_seen)

    # turn partitions into an indicator
    slice_indicator = np.ones(y.shape[0], dtype=np.int)
    for j, (start_idx, end_idx) in enumerate(
            zip(slice_partition, slice_partition[1:])):

        # this just puts any remaining observations in the last slice
        if j == len(slice_partition) - 2:
            slice_indicator[start_idx:] = j
        else:
            slice_indicator[start_idx:end_idx] = j

    slice_counts = np.bincount(slice_indicator)
    return slice_indicator, slice_counts
