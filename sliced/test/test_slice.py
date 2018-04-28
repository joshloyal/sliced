from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from sliced.datasets import load_athletes
from sliced.base import slice_y, grouped_sum


module_rng = np.random.RandomState(123)


def test_group_sum():
    """Simple group sum with a known result."""
    X = np.arange(25).reshape(5, 5)
    groups = np.array([0, 0, 1, 1, 2])

    expected_sum = np.array([
        [5, 7, 9, 11, 13],
        [25, 27, 29, 31, 33],
        [20, 21, 22, 23, 24]])
    np.testing.assert_array_equal(
        grouped_sum(X, groups),
        expected_sum
    )


def test_group_sum_all_same():
    """Test behavior when every row is in the same group."""
    X = np.arange(9).reshape(3, 3)
    groups = np.zeros(3)

    expected_sum = np.sum(X, axis=0).reshape(1, -1)
    np.testing.assert_array_equal(
        grouped_sum(X, groups),
        expected_sum
    )


def test_group_sum_all_different():
    """Test behavior when every row is in a different group."""
    X = np.arange(9).reshape(3, 3)
    groups = np.arange(3)
    np.testing.assert_array_equal(
        grouped_sum(X, groups),
        X
    )


def test_group_sum_too_many_groups():
    """Throw an error if too many groups are used."""
    X = np.arange(9).reshape(3, 3)
    groups = np.arange(4)
    with pytest.raises(IndexError):
        grouped_sum(X, groups)


def test_slice_raises_error_when_y_has_one_value():
    """Error if y only has one value."""
    y = np.ones(100)

    with pytest.raises(ValueError):
        slice_y(y)


def test_slice_y_simple():
    """Test a simple a target with 100 unique values and even slices."""
    shuffle = module_rng.permutation(100)
    y = np.arange(100)[shuffle]

    slice_indicator, counts = slice_y(y, n_slices=10)

    # all slices should have 10 counts
    np.testing.assert_array_equal(
        counts, np.repeat(10, 10))

    # continguous indices counting up from 10
    np.testing.assert_array_equal(
        slice_indicator,
        np.repeat(np.arange(10), 10)
    )


def test_uneven_slices():
    """Too match DR the slice function spills over to 6 slices if it cannot
    fit it into 5..."""
    shuffle = module_rng.permutation(68)
    y = np.arange(68)[shuffle]

    slice_indicator, counts = slice_y(y, n_slices=5)

    np.testing.assert_array_equal(
        counts,
        np.array([13, 13, 13, 13, 13, 3])
    )

    np.testing.assert_array_equal(
        slice_indicator,
        np.hstack((np.repeat(np.arange(5), 13), np.array([5, 5, 5])))
    )


def test_uneven_slices_within_y():
    """Make sure the slice function actually groups common y values together.
    """
    y = np.hstack((
        np.repeat(1, 13),
        np.repeat(2, 5),
        np.repeat(3, 25)))

    slice_indicator, counts = slice_y(y, n_slices=3)

    np.testing.assert_array_equal(
        counts, np.array([13, 5, 25]))

    expected_indicator = np.hstack((
        np.repeat(0, 13),
        np.repeat(1, 5),
        np.repeat(2, 25))
    )
    np.testing.assert_array_equal(
        slice_indicator, expected_indicator)


def test_remaining_obs_put_in_last_slice():
    """This was a bug found in testing. This particular dataset puts 9
    counts"""
    y = np.arange(100)

    slice_indicator, counts = slice_y(y, n_slices=11)

    np.testing.assert_array_equal(
        np.hstack((np.repeat(9, 10), 10)),
        counts
    )

    expected_slices = np.hstack((
        np.repeat(np.arange(10), 9),
        np.repeat(10, 10)
    ))
    np.testing.assert_array_equal(
        expected_slices,
        slice_indicator
    )


def test_slice_match_athletes():
    """Test that we match the slices used in the DR package on the
    athlete's dataset.
    """
    X, y = load_athletes()

    slice_indicator, counts = slice_y(y, n_slices=11)

    # counts per slice match DR package
    expected_counts = np.array([
        18, 18, 18, 18, 18, 19, 18, 19, 23, 18, 15])
    np.testing.assert_array_equal(counts, expected_counts)

    # slice indicators match DR package
    expected_indicator = np.array([
        6, 5, 4, 4, 2, 3, 5, 1, 3, 3, 8, 6, 1, 7, 5, 4, 3, 4, 6,
        4, 6, 6, 4, 3, 7, 6, 5, 4, 1, 2, 2, 5, 5, 6, 6, 2, 1, 3,
        2, 4, 4, 2, 3, 4, 2, 5, 4, 5, 3, 1, 3, 4, 3, 5, 5, 3, 3,
        3, 4, 2, 5, 5, 6, 5, 4, 2, 2, 2, 5, 8, 6, 6, 7, 5, 8, 1,
        3, 1, 3, 1, 3, 1, 2, 3, 2, 1, 2, 2, 2, 4, 4, 1, 2, 1, 2,
        1, 1, 1, 1, 1, 5, 8, 9, 10, 9, 8, 8, 9, 9, 10, 7, 9, 11, 9,
        9, 9, 11, 9, 10, 10, 1, 10, 10, 10, 10, 11, 10, 9, 11, 8, 10, 10, 11,
        11, 11, 9, 10, 8, 9, 8, 4, 7, 7, 8, 11, 10, 9, 6, 7, 8, 6, 5,
        4, 8, 3, 7, 7, 9, 9, 11, 9, 9, 11, 7, 9, 8, 7, 6, 6, 6, 7,
        7, 6, 6, 5, 11, 11, 11, 10, 7, 8, 9, 7, 9, 8, 8, 9, 7, 11, 9,
        9, 11, 8, 10, 10, 10, 7, 10, 8, 7, 6, 8,
    ])
    np.testing.assert_array_equal(
        slice_indicator, np.sort(expected_indicator) - 1)
