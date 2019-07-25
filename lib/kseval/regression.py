import numpy as np

from math import log, pi, sqrt
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import ndtr


LOG2PI2 = log(2 * pi) / 2


class DynamicLinearRegression:

    def __init__(self, *, time_kern, noise_var):
        self._time_kern = time_kern
        self._noise_var = noise_var
        self._X_train = None
        self._t_train = None

    def kernel(self, X1, t1, X2, t2):
        return self._time_kern.k_mat(t1, t2) * np.dot(X1, X2.T)

    def fit(self, X, t, y):
        self._X_train = X
        self._t_train = t
        n = len(y)
        K = self.kernel(X, t, X, t)
        # Compute lower triangular Cholesky factor.
        L = np.linalg.cholesky(K + self._noise_var * np.eye(n))
        # 'True' because L is lower triangular.
        alpha = cho_solve((L, True), y, check_finite=False)
        # Log-marginal likelihood.
        lml = -0.5*np.dot(y, alpha) - np.sum(np.log(np.diag(L))) - n*LOG2PI2
        # Store for prediction.
        self._L = L
        self._alpha = alpha
        return lml

    def predict(self, X, t):
        # Covariance between training set and test points.
        K = self.kernel(self._X_train, self._t_train, X, t)
        # Predictive mean.
        m = np.dot(K.T, self._alpha)
        # Predictive variance.
        v = solve_triangular(self._L, K, lower=True)
        var = self.kernel(X, t, X, t) - np.dot(v.T, v) + self._noise_var
        return m, np.diag(var)

    def probabilities(self, X, t):
        mu, var = self.predict(X, t)
        return ndtr(mu / np.sqrt(var))
