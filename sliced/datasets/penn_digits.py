import csv
import numpy as np

from os.path import dirname

from sliced.datasets.base import load_data


def load_penn_digits(digits='all', subset='train'):
    """Load and return the Penn Digits dataset (classification).

    This dataset concerns the identificatino of hand-written digits {0, 1, ..., 9}.
    It contains handwritten digits by 44 subjects, each of whom were asked to write
    250 random digits. Eight pairs of 2-dimensional locations were recorded on each digit,
    yielding a 16 dimensional feature vector. The 44 subjects are divided into two groups
    of size 30 and 14, in which the first form the training set (of size 7,494) and the
    second formed the test set (of sample size 3,498). The data set was taken from the
    UCI machine-learning repository at

    https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits

    =================   ==================
    Classes                             10
    Dimensionality                      16
    Features             integer, positive
    =================   ==================

    Parameters
    ----------
    digits : str or list (default='all')
        Which digits to include in the sample. The default is 'all' indicating
        all digits are included. Can be a list of digits for example [0, 6, 9]
        will only include digits 0, 6, and 9.

    subset : str {'train', 'test', 'all'}, (default='train')
        Which subset of the data to return. 'train' indicates the training set,
        'test' indicates the test set, and 'all' indicates a combination of
        both. Note that all does not shuffle the dataset, so the first samples
        are training and the remaining samples are test.
    """
    module_path = dirname(__file__)

    if subset not in ['train', 'test', 'all']:
        raise ValueError("`subset` must be one of {'train', 'test', 'all'}")

    if subset == 'train':
        data, target = load_data(module_path, 'pendigits.tra')
    elif subset == 'test':
        data, target = load_data(module_path, 'pendigits.tes')
    else:
        data_train, target_train = load_data(module_path, 'pendigits.tra')
        data_test, target_test = load_data(module_path, 'pendigits.tes')
        data = np.vstack((data_train, data_test))
        target = np.concatenate((target_train, target_test))

    if isinstance(digits, list):
        digit_mask = np.isin(target, digits)
        return data[digit_mask], target[digit_mask]

    return data, target

