from unittest.mock import MagicMock
import warnings

import numpy as np
import pytest
import sklearn
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline

import sklearn_expansion
from sklearn_expansion.linear_model import (
    LinearRegression,
    QuantileRegression,
    Lasso,
    QuantileLasso,
    AdaptiveLasso,
    QuantileAdaptiveLasso,
    GroupLasso,
    QuantileGroupLasso,
    SparseGroupLasso,
    QuantileSparseGroupLasso,
    AdaptiveSparseGroupLasso,
    QuantileAdaptiveSparseGroupLasso,
)


@pytest.mark.parametrize(
    "estimator",
    [
        LinearRegression(),
        QuantileRegression(),
        Lasso(),
        QuantileLasso(),
        AdaptiveLasso(),
        QuantileAdaptiveLasso(),
        GroupLasso(),
        QuantileGroupLasso(),
        SparseGroupLasso(),
        QuantileSparseGroupLasso(),
        AdaptiveSparseGroupLasso(),
        QuantileAdaptiveSparseGroupLasso(),
    ],
)
def test_model_fit_predict(estimator) -> None:
    X, y = make_regression()
    estimator.fit(X, y)
    estimator.predict(X)


@pytest.mark.parametrize(
    "estimator",
    [
        GroupLasso(),
        QuantileGroupLasso(),
        SparseGroupLasso(),
    ],
)
def test_group_model_fit_predict(estimator) -> None:
    X, y = make_regression()
    group_index = np.random.randint(low=0, high=5, size=X.shape[1])
    estimator.fit(X, y, group_index=group_index)
    estimator.predict(X)


@pytest.mark.parametrize("estimator", [AdaptiveLasso()])
def test_model_cross_validate(estimator) -> None:
    X, y = make_regression()
    cross_validate(estimator, X, y)


@pytest.mark.parametrize("estimator", [AdaptiveLasso()])
def test_model_pipeline_fit_predict(estimator) -> None:
    X, y = make_regression()
    pipe = Pipeline([("selector", SelectFromModel(Ridge())), ("estimator", estimator)])
    pipe.fit(X, y)


@pytest.mark.parametrize("estimator", [AdaptiveLasso()])
def test_model_pipeline_cross_validate(estimator) -> None:
    X, y = make_regression()
    pipe = Pipeline([("selector", SelectFromModel(Ridge())), ("estimator", estimator)])
    cross_validate(pipe, X, y)


@pytest.mark.parametrize("estimator", [AdaptiveLasso()])
@pytest.mark.parametrize("param_grid", [{"alpha": [0.01, 0.1, 1.0]}])
def test_model_pipeline_gridsearchcv_fit_predict(estimator, param_grid) -> None:
    X, y = make_regression()
    gscv = GridSearchCV(estimator, param_grid=param_grid)
    gscv.fit(X, y)
    best_estimator_ = gscv.best_estimator_
    best_estimator_.fit(X, y)


@pytest.mark.parametrize(
    "estimator",
    [
        LinearRegression(),
        QuantileRegression(),
        Lasso(),
        QuantileLasso(),
        AdaptiveLasso(),
        QuantileAdaptiveLasso(),
        GroupLasso(),
        QuantileGroupLasso(),
        SparseGroupLasso(),
        QuantileSparseGroupLasso(),
        AdaptiveSparseGroupLasso(),
        QuantileAdaptiveSparseGroupLasso(),
    ],
)
@pytest.mark.parametrize("param_grid", [{"alpha": [0.01, 0.1, 1.0]}])
def test_model_pipeline_gridsearchcv_cross_validate(estimator, param_grid) -> None:
    X, y = make_regression()
    gscv = GridSearchCV(estimator, param_grid=param_grid)
    cross_validate(gscv, X, y)


@pytest.mark.parametrize("alpha", [1e-10, 1e-5, 1e-0, 0, 1e1])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_compare_lasso(alpha, fit_intercept) -> None:
    X, y = make_regression()
    sklasso = sklearn.linear_model.Lasso(alpha=alpha, fit_intercept=fit_intercept)
    lasso = sklearn_expansion.linear_model.Lasso(
        alpha=alpha, fit_intercept=fit_intercept
    )
    sklasso.fit(X, y)
    lasso.fit(X, y)

    y_sk_predict = sklasso.predict(X)
    y_predict = lasso.predict(X)

    corrcoef = np.corrcoef(y_sk_predict, y_predict)
    assert corrcoef[0][1] >= 0.97
