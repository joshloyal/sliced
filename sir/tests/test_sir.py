import numpy as np

from sir import SlicedInverseRegression
from sir import datasets


def test_sir():
    X, y = datasets.make_cubic(random_state=123)

    sir = SlicedInverseRegression().fit(X, y)

    true_beta = (1 / np.sqrt(2)) * np.hstack((np.ones(2), np.zeros(8)))
    angle = np.dot(true_beta, sir.vectors_[:, 0])
    print(angle)
    np.testing.assert_allclose(np.abs(angle), 1, rtol=1e-2)
