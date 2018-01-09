from sklearn.utils.estimator_checks import check_estimator

from rdlearn import SlicedInverseRegression, SlicedAverageVarianceEstimation


def test_sir():
    check_estimator(SlicedInverseRegression)


def test_save():
    check_estimator(SlicedAverageVarianceEstimation)
