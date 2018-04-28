.. -*- mode: rst -*-

|Travis|_ |AppVeyor|_ |Coveralls|_ |CircleCI|_ |License|_

.. |Travis| image:: https://travis-ci.org/joshloyal/sliced.svg?branch=master
.. _Travis: https://travis-ci.org/joshloyal/sliced

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/54j060q1ukol1wnu/branch/master?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/joshloyal/sliced/history

.. |Coveralls| image:: https://coveralls.io/repos/github/joshloyal/sliced/badge.svg?branch=master
.. _Coveralls: https://coveralls.io/github/joshloyal/sliced?branch=master

.. |CircleCI| image:: https://circleci.com/gh/joshloyal/sliced/tree/master.svg?style=svg
.. _CircleCI: https://circleci.com/gh/joshloyal/sliced/tree/master

.. |License| image:: https://img.shields.io/badge/License-MIT-blue.svg
.. _License: https://opensource.org/licenses/MIT

.. _scikit-learn: https://github.com/scikit-learn/scikit-learn

sliced
======
sliced is a python package offering a number of sufficient dimension reduction (SDR) techniques commonly used in high-dimensional datasets with a supervised target. It is compatible with scikit-learn_.


Algorithms supported:

- Sliced Inverse Regression (SIR) [1]_
- Sliced Average Variance Estimation (SAVE) [2]_

Documentation / Website: https://joshloyal.github.io/sliced/


Example
-------
Example that shows how to learn a one dimensional subspace from a dataset with ten features:

.. code-block:: python

   from sliced.datasets import make_cubic
   from sliced import SlicedInverseRegression

   # load the 10-dimensional dataset
   X, y = make_cubic(random_state=123)

   # Set the options for SIR
   sir = SlicedInverseRegression(n_directions=1)

   # fit the model
   sir.fit(X, y)

   # transform into the new subspace
   X_sir = sir.transform(X)


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
You need a working installation of numpy and scipy to install sliced. If you have a working installation of numpy and scipy, the easiest way to install sliced is using ``pip``::

    pip install -U sliced

If you prefer, you can clone the repository and run the setup.py file. Use the following commands to get the copy from GitHub and install all the dependencies::

    git clone https://github.com/joshloyal/sliced.git
    cd sliced
    pip install .

Or install using pip and GitHub::

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
