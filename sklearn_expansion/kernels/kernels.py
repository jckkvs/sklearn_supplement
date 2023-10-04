from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin, Kernel
import numpy as np

class SplineKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    def __init__(self):
        pass

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X

        K = self._spline_kernel(X, Y)
        
        if eval_gradient:
            # Gradient is not supported
            raise ValueError("Gradient can't be computed for Spline Kernel.")

        return K

    def diag(self, X):
        return np.array([1.0] * X.shape[0])  # 正規化されたカーネルの対角要素は1

    def is_stationary(self):
        return True  # スプラインカーネルは定常

    def _spline_kernel(self, X, Y):
        min_xy = np.minimum(X, Y.T)
        return 1 + X @ Y.T + X @ Y.T * min_xy - ((X + Y.T) / 2) * min_xy ** 2 + (min_xy ** 3) / 3

class ARDKernel(StationaryKernelMixin, Kernel):
    def __init__(self, length_scale):
        self.length_scale = np.array(length_scale)

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X

        dists = np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2 / self.length_scale, axis=-1)
        K = np.exp(-0.5 * dists)

        return K
class SigmoidKernel(Kernel):
    def __init__(self, alpha=1.0, c=0.0):
        self.alpha = alpha
        self.c = c

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        K = np.tanh(self.alpha * np.dot(X, Y.T) + self.c)
        return K
class PolynomialKernel(Kernel):
    def __init__(self, d=3, c=1):
        self.d = d
        self.c = c

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        K = (np.dot(X, Y.T) + self.c) ** self.d
        return K

class HistogramKernel(Kernel):
    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        K = np.minimum(X[:, np.newaxis], Y[np.newaxis, :])
        return K
class ChebyshevKernel(Kernel):
    def __init__(self, lambda_param=1.0):
        self.lambda_param = lambda_param

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        dists = np.max(np.abs(X[:, np.newaxis] - Y[np.newaxis, :]), axis=-1)
        K = 1.0 / np.sqrt(1 + self.lambda_param * dists)
        return K

class CosineSimilarityKernel:
    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        X_norm = np.linalg.norm(X, axis=1)
        Y_norm = np.linalg.norm(Y, axis=1)
        return np.dot(X, Y.T) / (X_norm[:, np.newaxis] * Y_norm[np.newaxis, :])

class LaplacianKernel:
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        dists = np.sum(np.abs(X[:, np.newaxis] - Y[np.newaxis, :]), axis=2)
        return np.exp(-dists / self.sigma)

class ChiSquaredKernel:
    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        dists = np.sum((X[:, np.newaxis] - Y[np.newaxis, :]) ** 2 / (X[:, np.newaxis] + Y[np.newaxis, :] + 1e-10), axis=2)
        return np.exp(-dists)
class LogisticKernel:
    def __init__(self, alpha=1.0, c=0.0):
        self.alpha = alpha
        self.c = c

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        return np.log(1 + np.exp(self.alpha * np.dot(X, Y.T) + self.c))
class PerceptronKernel:
    def __init__(self, p=1):
        self.p = p

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        return (np.dot(X, Y.T) + 1) ** self.p


