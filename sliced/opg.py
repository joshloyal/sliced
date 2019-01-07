from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import numpy as np
import scipy.linalg as linalg

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import normalize, StandardScaler

from .externals import check_array, check_X_y, check_is_fitted
from .base import is_multioutput


__all__ = ['OuterProductGradients']


class OuterProductGradients(BaseEstimator, TransformerMixin):
    """Outer Product of Gradients (OPG) [1]

    Forward regression methods estimate the span of the
    central mean subspace S_{E[y|X]} using ideas from more familiar
    estimates of the regression function E[y|X]. The advantage of these methods
    is that they do not need to place distribution assumptions on the predictors
    X such as SIR or SAVE. However, they rely on high dimension kernel functions
    which are especially in-efficient for small sample sizes. In addition,
    they assume the response y is continous.

    The Outer Product of Gradients method of Xia et al (2002) takes advantage
    of the fact that span{E[grad(E[y|X])grad(E[y|X])^T]} = S_{E[y|X]}. In
    order to estimate this span, the OPG method estimates the local gradient
    using local linear regression. The sample expectation of the outer
    product of estimated gradients is formed, and the spectral decomposition
    of this product is used to estimated the span of the central mean subspace.

    Parameters
    ----------
    n_directions : int, str or None (default='auto')
        Number of directions to keep. Corresponds to the dimension of
        the central mean subpace. If n_directions==None,
        the number of directions equals the number of features.

    kernel : string or callable, default='rbf'
        Kernel mapping used internally. A callable should accept two arguments
        and the keyword arguments passed to his object as kernel_params, and
        should return a floating point number.

    gamma : float, default='auto'
        Gamma parameter for the RBF, laplacian, polynomial, exponential, chi2,
        and sigmoid kernels. If gamma='auto, the default for the RBF kernel
        in Xia et al. is used.

    degree : float, default=2
        Degree of the polynomial kernel. Ignored by other kernels.

    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : mapping of string to any, optional
        Additional parameters (keyword arguments) for a kernel function
        passed as a callable object.

    copy : bool (default=True)
         If False, data passed to fit are overwritten and running
         fit(X).transform(X) will not yield the expected results,
         use fit_transform(X) instead.

    Attributes
    ----------
    directions_ : array, shape (n_directions, n_features)
        The directions in feature space, representing the
        central mean subspace which is sufficient to describe the conditional
        mean E[y|X]. The directions are sorted by ``eigenvalues_``.

    eigenvalues_ : array, shape (n_directions,)
        The eigenvalues corresponding to each of the selected directions.
        These are the eigenvalues of the expected outer product of gradients.
        Larger eigenvalues indicate more prevalent directions.

    References
    ----------

    [1] Xia, Y., Tong, H., Li, W. K., and Zhu, L.-X. (2002),
        “An adaptive estimation of di- mension reduction space,”
        Journal of the Royal Statistical Society: Series B (Statistical Methodology),
        64, 363–410.
    """
    def __init__(self,
                 n_directions=None,
                 kernel='rbf',
                 gamma='auto',
                 degree=2,
                 coef0=1,
                 kernel_params=None,
                 copy=True):
        self.n_directions = n_directions
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.copy = copy

    def _get_gamma(self, X):
        n_samples, n_features = X.shape
        if isinstance(self.gamma, six.string_types):
            # bandwidth proposed in Xia (2007)
            gamma = 2.34 * n_samples ** (-1 /
                                          (np.maximum(n_features, 3) + 6))
            return 1 / (gamma * 2)
        return self.gamma

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self._get_gamma(X),
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def fit(self, X, y, kernel=None):
        """Fit the model with X and y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target values (class labels in classification, real numbers
            in regression).

        kernel : array-like, shape (n_samples, n_samples)
            Precomputed kernel matrix.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if sparse.issparse(X):
            raise TypeError("OuterProductGradient does not support "
                            "sparse input.")

        X, y = check_X_y(X, y, dtype=[np.float64, np.float32],
                         y_numeric=True, copy=self.copy)

        n_samples, n_features = X.shape

        self.n_directions_ = (n_features if self.n_directions is None else
                              self.n_directions)

        # validate y
        if is_multioutput(y):
            raise TypeError("The target `y` cannot be multi-output.")

        # Center and scale feature matrix
        scaler = StandardScaler(copy=self.copy)
        X = scaler.fit_transform(X)

        # pre-compute the kernel weights
        K = self._get_kernel(X) if kernel is None else kernel

        # calculate the gradient estimate through a local wls
        gradients = np.zeros((n_features, n_samples))
        for i in range(n_samples):
            linear_model = LinearRegression(fit_intercept=True)
            linear_model.fit(X - X[i, :], y, sample_weight=K[i, :])
            gradients[:, i] = linear_model.coef_

        # solve eigen-value problem, i.e. outer product of gradient estimates
        M = np.dot(gradients, gradients.T)
        evals, evecs = linalg.eigh(M)
        evecs = evecs[:, ::-1]
        evals = evals[::-1]

        # convert back to un-scaled basis
        directions = evecs / np.sqrt(scaler.var_)
        directions = normalize(
            directions[:, :self.n_directions_], norm='l2', axis=0)
        self.directions_ = directions.T
        self.eigenvalues_ = evals[:self.n_directions_]

        return self

    def transform(self, X):
        """Apply dimension reduction on X.

        X is projected onto the central subspace  previously extracted from a
        training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.


        Returns
        -------
        X_new : array-like, shape (n_samples, n_directions)

        """
        check_is_fitted(self, 'directions_')

        X = check_array(X)
        return np.dot(X, self.directions_.T)
