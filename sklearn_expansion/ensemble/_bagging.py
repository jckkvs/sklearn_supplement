from sklearn.base import RegressorMixin
from sklearn.ensemble import BaggingRegressor
import numpy as np

class MultiRegressor(BaggingRegressor):
    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator

    def fit(self, X, y, group=None):
        self.estimator = self.estimator.fit(X,y)
        return self

    
    def predict(self,X, return_std=False):
        if return_std == True:
            if hasattr(self.estimator,"estimators_") == False:
                raise ValueError(f"{self.estimator} has no attribute estimators_")
            
            y_predicts = []
            for each_estimator in self.estimator.estimators_:
                y_predicts.append(each_estimator.predict(X))

            y_predicts_np = np.array(y_predicts)
            y_std = np.std(y_predicts_np, axis=0)
            return self.estimator.predict(X), y_std

        else:
            return self.estimator.predict(X)