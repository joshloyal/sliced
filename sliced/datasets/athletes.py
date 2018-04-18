import csv
from os.path import dirname

from sliced.datasets.base import load_data


def load_athletes():
    """Load and return the australian athletes data set (regression).

    This dataset was collected in a study of how data on various
    characteristics of the blood varied with sport body size and sex of the
    athlete.

    Here the goal is to predict lean body mass (lbm) based on the logarithm
    of various attributes of the athelete.

    ===============     ===============
    Samples total                   202
    Dimensionality                    8
    Features             real, positive
    ===============     ===============

    Returns
    -------
    (data, target) : tuple
        The X feature matrix and the y target vector.

    References
    ----------

    [1] Telford, R.D. and Cunningham, R.B. 1991.
        "Sex, sport and body-size dependency of hematology in highly
        trained athletes", Medicine and Science in Sports and
        Exercise 23: 788-794.
    """
    module_path = dirname(__file__)
    data, target = load_data(module_path, 'athletes.csv',
                             is_classification=False)
    return data, target
