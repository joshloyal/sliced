import csv
from os.path import dirname

from sliced.datasets.base import load_data


def load_banknote():
    """Load and return the swiss banknote dataset (classification).

    Six measurements made on 100 genuine Swiss banknotes and 100 counterfeit
    ones.

    Features
    ========
    Length:
        Length of bill (mm)
    Left:
        Width of left edge (mm)
    Right:
        Width of right edge (mm)
    Bottom:
        Bottom margin width (mm)
    Top:
        Top margin width (mm)
    Diagonal:
        Length of image diagonal (mm)
    Y:
        0 = genuine, 1 = conterfeit

    =================   =================
    Classes                             2
    Samples per class    100 (Y), 100 (N)
    Samples total                     200
    Dimensionality                      6
    Features               real, positive
    =================   =================
    """
    module_path = dirname(__file__)
    data, target = load_data(module_path, 'banknote.csv')
    return data, target
