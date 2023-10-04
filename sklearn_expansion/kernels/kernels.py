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
