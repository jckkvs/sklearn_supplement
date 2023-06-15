# Authors: Yuki Horie(2021)

# Ref.1 https://hktech.hatenablog.com/entry/2018/10/05/004235

import copy
import traceback
import pprint
import random
import importlib

import numbers
import numpy as np
import pandas as pd
from scipy.stats import f, chi2
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import model_selection

from sklearn.feature_selection import SelectorMixin
from sklearn.base import BaseEstimator, clone, MetaEstimatorMixin

from sklearn.exceptions import NotFittedError
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.ensemble import RandomForestRegressor

class RegularizedRandomForestRegressor(MetaEstimatorMixin, SelectorMixin, BaseEstimator):

    def __init__(self,
                estimator=RandomForestRegressor(),
                n_estimators=100,
                *,
                criterion="squared_error",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features=1.0,
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=True,
                oob_score=False,
                n_jobs=None,
                random_state=None,
                verbose=0,
                warm_start=False,
                ccp_alpha=0.0,
                max_samples=None,):

        self.estimator = estimator
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples


    def _get_support_mask(self):
        # SelectByModel can directly call on transform.
    
        if self.prefit:
            estimator = self.estimator
        elif hasattr(self, 'estimator_'):            
            estimator = self.estimator_
        else:
            raise ValueError('Either fit the model before transform or set'
                             ' "prefit=True" while passing the fitted'
                             ' estimator to the constructor.')

        mask = _get_selected(self.scores, self.threshold, self.min_features, self.max_features)

        if not mask.any():
            warn(
                "No features were selected: either the data is"
                " too noisy or the selection test too strict.",
                UserWarning,
            )
            return np.empty(0).reshape((X.shape[0], 0))
        if len(mask) != self.X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")

        return mask

    def fit(self, X, y=None, **fit_params):
        """Fit the SelectFromModel meta-transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        **fit_params : Other estimator specific parameters

        Returns
        -------
        self : object
        """
        self.X = X
        self.y = y
        if self.max_features is not None:
            if not isinstance(self.max_features, numbers.Integral):
                raise TypeError("'max_features' should be an integer between"
                                " 0 and {} features. Got {!r} instead."
                                .format(X.shape[1], self.max_features))
            elif self.max_features < 0 or self.max_features > X.shape[1]:
                raise ValueError("'max_features' should be 0 and {} features."
                                 "Got {} instead."
                                 .format(X.shape[1], self.max_features))

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)
            
        feature_importances_ = self.estimator_.feature_importances_

        for t in range(self.n_estimators):
            pass


        min_features = self.min_features
        max_features = self.max_features
        support_ = _get_selected(scores, threshold, min_features, max_features)
        self.support_ = support_

        return self

    @property
    def threshold_(self):
        scores = _get_scores(self.estimator_,  self.selector_method, self.norm_order)
        return _calculate_threshold(self.estimator, self.X, self.y,  self.selector_method, scores, self.threshold, self.cv)

    @if_delegate_has_method('estimator')
    def partial_fit(self, X, y=None, **fit_params):
        """Fit the SelectFromModel meta-transformer only once.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        **fit_params : Other estimator specific parameters

        Returns
        -------
        self : object
        """
        if self.prefit:
            raise NotFittedError(
                "Since 'prefit=True', call transform directly")
        if not hasattr(self, "estimator_"):
            self.estimator_ = clone(self.estimator)
        self.estimator_.partial_fit(X, y, **fit_params)
        return self

    def _more_tags(self):
        estimator_tags = self.estimator._get_tags()
        return {'allow_nan': estimator_tags.get('allow_nan', True)}
