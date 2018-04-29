.. currentmodule:: sliced

sliced API
==========
Major classes are :class:`sir.SlicedInverseRegression` and
:class:`save.SlicedAverageVarianceEstimation`.

Transformers
------------
.. autosummary::
    :toctree: generated/

    sir.SlicedInverseRegression
    save.SlicedAverageVarianceEstimation


Datasets
--------

Datasets used in the examples as well as to test the algorithms are contained in
the datasets module.

.. autosummary::
    :toctree: generated/

    datasets.make_cubic
    datasets.make_quadratic
    datasets.make_polynomial


Utilities
---------

.. autosummary::
    :toctree: generated/

    base.grouped_sum
    base.slice_y
