.. -*- mode: rst -*-
|Travis|_ |AppVeyor|_ |Coveralls|_ |CircleCI|_

.. |Travis| image:: https://travis-ci.org/joshloyal/sliced.svg?branch=master
.. _Travis: https://travis-ci.org/joshloyal/sliced

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/54j060q1ukol1wnu/branch/master?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/joshloyal/sliced/history

.. |Coveralls| image:: https://coveralls.io/repos/github/joshloyal/sliced/badge.svg?branch=master)
.. _Coveralls: https://coveralls.io/github/joshloyal/sliced?branch=master

.. |CircleCI| image:: https://circleci.com/gh/joshloyal/sliced/tree/master.svg?style=svg
.. _CircleCI: https://circleci.com/gh/joshloyal/sliced/tree/master

.. _scikit-learn: https://github.com/scikit-learn/scikit-learn

sliced
======
sliced is a python package offering a number of sufficient dimension reduction (SDR) techniques commonly used in high-dimensional datasets with a supervised target. It is compatible with scikit-learn_.

The algorithms available in sliced include:

- Sliced Inverse Regression (SIR) [1]_
- Sliced Average Variance Estimation (SAVE) [2]_

Website: https://joshloyal.github.io/sliced/


Installation
------------

Dependencies
------------
sliced requires:

- Python (>= 2.7 or >= 3.4)
- NumPy (>= 1.8.2)
- SciPy (>= 0.13.3)
- Scikit-learn (>=0.17)

Additionally, to run examples, you need matplotlib(>=2.0.0).

Installation
------------
You need a working installatio of numpy and scipy to install sliced. If you have a working installation of numpy and sliced, the easiest way to install sliced is using ``pip``::

    pip install -U sliced

If you prefer, you can clone the repository and run the setup.py file. Use the following commands to get the copy from GitHub and install all the dependencies::

    git clone https://github.com/joshloyal/sliced.git
    cd sliced
    pip install .

Of install using pip and GitHub::

    pip install -U git+https://github.com/joshloyal/sliced.git

Testing
-------
After installation, you can use pytest to run the test suite via setup.py::

    python setup.py test

References:
-----------
.. [1] : Li, K C. (1991)
        "Sliced Inverse Regression for Dimension Reduction (with discussion)",
        Journal of the American Statistical Association, 86, 316-342.
.. [2] : Shao, Y, Cook, RD and Weisberg, S (2007).
         "Marginal Tests with Sliced Average Variance Estimation",
         Biometrika, 94, 285-296.
