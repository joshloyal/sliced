from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.utils.estimator_checks import check_estimator

from sliced import SlicedInverseRegression, SlicedAverageVarianceEstimation


def test_sir():
    # a lot of warnings about changing n_slices for classification problems
    check_estimator(SlicedInverseRegression)


def test_save():
    # a lot of warnings about changing n_slices for classification problems
    check_estimator(SlicedAverageVarianceEstimation)
