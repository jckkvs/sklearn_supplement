from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from lineartree import LinearTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import shap
from sklearn.base import clone
import numpy as np
import warnings

class LinearTreeSHAPRegressor(BaseEstimator):
    '''
    A lineartree SHAP Values regressor.
    A model that predicts SHAP values obtained by a nonlinear machine learning model using a linear tree model for each explanatory variable.

    Parameters
    ----------
    estimator : Non-linear machine learning model
        A machine learning estimator that is fit to all data and used to calculate the shap value.

    shapvalue_estimator : LinearTreeRegressor(estimator=LinearRegression())
        A LinearTreeModel that learns the relationship between each explanatory variable and the corresponding shap_value.

    Attriutes
    ----------
    estimator_ : DecisionTreeRegressor
        The estimator fitted to all-data.

    shap_values_estimators_ : list of LinearTreeRegressors

    predict_shap_values_ : list of each shap values predicted by shap_values_models

    shap_values_sum_ : base shap values.

    '''
    def __init__(self,
                estimator=RandomForestRegressor(),
                shapvalue_estimator=LinearTreeRegressor(base_estimator=LinearRegression(),max_depth=1),
                *,
                base_estimator="deprecated"):
        self.estimator = estimator
        self.shapvalue_estimator = shapvalue_estimator
        self.base_estimator = base_estimator

    def fit(self, X, y):

        if self.base_estimator != "deprecated":
            warnings.warn(
                "`base_estimator` was renamed to `estimator` in version 0.3.6 and "
                "will be removed in 1.0",
                FutureWarning,
            )
            self.estimator = self.base_estimator


        self.estimator.fit(X,y)
        self.explainer = shap.TreeExplainer(model=self.estimator)
        self.shap_values = self.explainer.shap_values(X=X)
        shap_values_estimators_ = []
        for each_X, shap_value in zip(X.T, self.shap_values.T):
            # print(shap_value)
            # print(shap_value.shape)
            each_X = each_X.reshape(-1,1)
            each_ltr = clone(self.shapvalue_estimator)
            each_ltr.fit(each_X, shap_value)
            shap_values_estimators_.append(each_ltr)
        self.shap_values_estimators_ = shap_values_estimators_

        return self

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for idx, each_shap_value_estimator_ in enumerate(self.shap_values_estimators_):
            each_pred_shap_value = each_shap_value_estimator_.predict(X[:, idx].reshape(-1,1))
            y_pred += each_pred_shap_value
        shap_values_pred = self.explainer.shap_values(X=X)
        y_pred += shap_values_pred.sum(axis=1)
        return y_pred