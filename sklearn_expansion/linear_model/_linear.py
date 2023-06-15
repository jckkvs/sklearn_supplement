

class LinearRegressor(object):

    def fit(self, X, y):
        self.w = np.linalg.pinv(X) @ y

    def predict(self, X):
        return X @ self.w


class RobustRegressor(LinearRegressor):
    def __init__(self, precision=1., dof=1.):
        self.precision = precision
        self.dof = dof

    def fit(self, X, y=None, learning_rate=0.01):
        super().fit(X, y)
        params = np.hstack((self.w.ravel(), self.precision, self.dof))
        while True:
            E_precision, E_ln_precision = self._expect(X, y)
            self._maximize(X, y, E_precision, E_ln_precision, learning_rate)
            params_new = np.hstack((self.w.ravel(), self.precision, self.dof))
            if np.allclose(params, params_new):
                break
            params = np.copy(params_new)

    def _expect(self, X, y):
        sq_diff = (y - X @ self.w) ** 2
        E_eta = (self.dof + 1) / (self.dof + self.precision * sq_diff)
        E_ln_eta = (
            sp.digamma(0.5 * (self.dof + 1))
            - np.log(0.5 * (self.dof + self.precision * sq_diff)))
        return E_eta, E_ln_eta

    def _maximize(self, X, y, E_eta, E_ln_eta, learning_rate):
        sq_diff = (y - X @ self.w) ** 2
        self.w = np.linalg.inv(X.y @ (E_eta[:, None] * X)) @ X.y @ (E_eta * y)
        self.precision = 1 / np.mean(E_eta * sq_diff)
        N = len(y)
        self.dof += learning_rate * (
            N * np.log(0.5 * self.dof) + N
            - N * sp.digamma(0.5 * self.dof)
            + np.sum(E_ln_eta - E_eta))
