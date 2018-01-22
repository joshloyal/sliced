from __future__ import print_function

import os
import sys

from setuptools import setup, find_packages


HERE = os.path.dirname(os.path.abspath(__file__))

# import ``__version__` from code base
exec(open(os.path.join(HERE, 'sliced', 'version.py')).read())


with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]


with open('test_requirements.txt') as f:
    TEST_REQUIRES = [l.strip() for l in f.readlines() if l]


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


DISTNAME = 'sliced'
DESCRIPTION = 'Toolbox for sufficient dimension reduction (SDR).'
with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Joshua D. Loyal'
MAINTAINER_EMAIL = 'jloyal25@gmail.com'
URL ='https://joshloyal.github.io/sliced'
DOWNLOAD_URL = 'https://pypi.org/project/sliced/#files'
LICENSE = 'MIT'
VERSION = __version__
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved :: MIT License',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 2',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
              ]


setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    zip_safe=False,
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require={'test': TEST_REQUIRES},
    setup_requires=['pytest-runner'],
    tests_require=TEST_REQUIRES,
)
