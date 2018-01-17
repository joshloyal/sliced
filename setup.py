from __future__ import print_function

import os
import sys

from setuptools import setup, find_packages


HERE = os.path.dirname(os.path.abspath(__file__))

# import ``__version__` from code base
exec(open(os.path.join(HERE, 'sliced', 'version.py')).read())


with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]


TEST_REQUIRES = [
    'pytest',
    'pytest-pep8',
    'nose>=1.1.2'
]


try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)


try:
    import scipy
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)


setup(
    name="sliced",
    version=__version__,
    description='Scikit-Learn compatabile Sufficient Dimension Reduction algorithms.',
    author='Joshua D. Loyal',
    author_email='jloyal25@gmail.com',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require={'test': TEST_REQUIRES},
    setup_requires=['pytest-runner'],
    tests_require=TEST_REQUIRES,
    test_suite='py.test',
    url='https://github.com/joshloyal/sliced',
    license='MIT',
)
