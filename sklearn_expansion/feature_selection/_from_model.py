# Authors: Gilles Louppe, Mathieu Blondel, Maheshakya Wijewardena
# Editor : Yuki Horie(2021)
# License: BSD 3 clause

# Ref.1 https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/cem.3226


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


def _calculate_threshold(estimator, X, y, selector_method, scores, threshold=None, percentile=None, magnification=1):
    """Interpret the threshold value"""

    if selector_method is not None:
        # Ref.1 https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/cem.3226
        if selector_method in ['LWs', 'LW','lws', 'lw', 'loading']:
            # Ref.1 3.1.1 LWs from PLS
            # median(w) / Interquartilerange(w)
            threshold = np.median(scores) / np.percentile(scores, 25)

        elif selector_method in ['RCs', 'rcs']:
            # Ref.1 3.1.2 Rcs from PLS
            # median(w) / Interquartilerange(w)
            threshold = np.median(scores) / np.percentile(scores, 25)


        elif selector_method in ['VIP']:
            # Ref.1 3.1.3 Variable importance in PLS projection
            if threshold is None:
                print('VIP need user-defined threshold')
                threshold = 0.5
                    
            else:
                threshold = threshold

        elif selector_method in ['SR']:
            # Ref.1 3.1.4 SR from PLS
            alpha = 0.32
            if threshold is None:
                n = self.X.shape[0]
                print(n)
                threshold = f.ppf(alpha, n-2, n-3)
                print(threshold)
                #raise ValueError('PLS-VIP needs the threshold as hyperparam')
                
            else:
                threshold = threshold

        elif selector_method in ['sMC', 'SMC']:
            # Ref.1 3.1.5 Significance multivariate correlation with PLS
            if threshold is None:
                raise ValueError('PLS-VIP needs the threshold as hyperparam')
                
            else:
                threshold = threshold

        elif selector_method in ['mRMR']:
            # Ref.1 3.1.5 Minimum redundancy maximum relevance in PLS
            if threshold is None:
                raise ValueError('PLS-VIP needs the threshold as hyperparam')
                
            else:
                threshold = threshold

        elif selector_method in ['GA-PLS','GAPLS','GA_PLS']:
            # Ref.1 3.2.1 Genetic algorithm combined with PLS regression
            if threshold is None:
                raise ValueError('PLS-VIP needs the threshold as hyperparam')
                
            else:
                threshold = threshold

        elif selector_method in ['SPA-PLS','PLS-SPA','SPAPLS','PLSSPA']:
            # Ref.1 3.2.2 Uninformative variable elimination in PLS
            if threshold is None:
                raise ValueError('PLS-VIP needs the threshold as hyperparam')
                
            else:
                threshold = threshold

        elif selector_method in ['SPA-PLS','PLS-SPA','SPA_PLS','PLS_SPA','SPAPLS','PLSSPA']:
            # Ref.1 3.2.3 Sub-window permutation analysis coupled with PLS
            if threshold is None:
                raise ValueError('PLS-VIP needs the threshold as hyperparam')
                
            else:
                threshold = threshold

        elif selector_method in ['PLS-IPW','IPW-PLS','PLSIPW','IPWPLS','PLS_IPW','IPW_PLS']:
            # Ref.1 3.2.4 Iterative predictor weighting PLS
            if threshold is None:
                raise ValueError('PLS-VIP needs the threshold as hyperparam')
                
            else:
                threshold = threshold

        elif selector_method in ['Backward']:
            # Ref.1 3.2.5 Backward variable elimination PLS
            if threshold is None:
                raise ValueError('PLS-VIP needs the threshold as hyperparam')
                
            else:
                threshold = threshold

        elif selector_method in ['REP']:
            # Ref.1 3.2.6 Regularized elimination procedure in PLS
            if threshold is None:
                raise ValueError('PLS-VIP needs the threshold as hyperparam')
                
            else:
                threshold = threshold

        elif selector_method in ['T2', 'Hotelling', 'Hotelling-T2', 'Hotelling_T2']:
            # Ref.1 3.2.7 Hotelling T2based variable selection in PLS 
            a_star = estimator.n_components

            if percentile is None:
                # default_selector_parameters
                percentile=0.01
                    
            else:
                percentile = percentile

            threshold = chi2.ppf(percentile, a_star) 

        elif selector_method in ['ST', 'STPLS']:
            # Ref.1 3.3.1 ST PLS
            threshold = threshold

        elif selector_method in ['TRUNC', 'STPLS']:
            # Ref.1 3.3.2 Distribution-based truncation for variable selection in PLS 
            threshold = threshold

        elif selector_method in ['PLS-WVC', 'PLS_WVC', 'PLSWVC']:
            # Ref.1 3.3.3 Weighted variable contribution in PLS 
            threshold = threshold
        
        elif selector_method == 'RFR':
            selector_method = None

        else:
            selector_method = None


        if threshold is not None:  
            print('threshold : ',threshold)      
            return threshold

        elif threshold is None:        
            print("{} is not supported to calculate".format(selector_method))
            print("Calculate the threshold, ignoring the given selector_method name {} ". format(selector_method))

        
    if selector_method is None:
        if threshold is None:
            # determine default from estimator
            est_name = estimator.__class__.__name__
            if ((hasattr(estimator, "penalty") and estimator.penalty == "l1") or
                    "Lasso" in est_name):
                # the natural default threshold is 0 when l1 penalty was used
                threshold = 1e-5
            else:
                threshold = "mean"

        if isinstance(threshold, str):
            if threshold == "median":
                threshold = np.median(scores)

            elif threshold == "mean":
                threshold = np.mean(scores)

            else:
                raise ValueError("Expected threshold='mean' or threshold='median' "
                                "got %s" % threshold)

        else:
            threshold = float(threshold)

    threshold = threshold * magnification

    return threshold


def _get_feature_importances(estimator, norm_order=1):
    """Retrieve or aggregate feature importances from estimator"""
    importances = getattr(estimator, "feature_importances_", None)

    coef_ = getattr(estimator, "coef_", None)
    if importances is None and coef_ is not None:
        if estimator.coef_.ndim == 1:
            importances = np.abs(coef_)

        else:
            importances = np.linalg.norm(coef_, axis=0,
                                         ord=norm_order)

    elif importances is None:
        raise ValueError(
            "The underlying estimator %s has no `coef_` or "
            "`feature_importances_` attribute. Either pass a fitted estimator"
            " to SelectFromModel or call fit before calling transform."
            % estimator.__class__.__name__)

    return importances


def _get_scores(estimator, selector_method, norm_order=1):

    """Retrieve or aggregate feature importances from estimator"""
    if selector_method is None:
        importances = getattr(estimator, "feature_importances_", None)
        coef_ = getattr(estimator, "coef_", None)
        if importances is None and coef_ is not None:
            if estimator.coef_.ndim == 1:
                scores = np.abs(coef_)

            else:
                scores = np.linalg.norm(coef_, axis=0,
                                            ord=norm_order)

        elif importances is None:
            raise ValueError(
                "The underlying estimator %s has no `coef_` or "
                "`feature_importances_` attribute. Either pass a fitted estimator"
                " to SelectFromModel or call fit before calling transform."
                % estimator.__class__.__name__)
        else:
            scores = importances

    elif selector_method in ['LWs', 'LW','lws', 'lw', 'loading']:
        # Ref.1 3.1.1 LWs from PLS
        x_loadings_ = getattr(estimator, "x_loadings_", None)
        if x_loadings_ is not None:
            scores = abs(x_loadings_).mean(axis=1)

    elif selector_method in ['RCs', 'rcs']:
        # Ref.1 3.1.2 Rcs from PLS
        coef_ = getattr(estimator, "coef_", None)
        if coef_ is not None:
            scores = abs(coef_).mean(axis=1)


    elif selector_method in ['VIP']:
        # Ref.1 3.1.3 Variable importance in PLS projection
        # X = T*P.T + E 
        # y = T*q   + f
        # T = X * W
        def vip(model):
            t = model.x_scores_
            w = model.x_weights_
            q = model.y_loadings_
            p, h = w.shape
            vips = np.zeros((p,))
            s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
            total_s = np.sum(s)
            for i in range(p):
                weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
                vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
            return vips
        scores = vip(estimator)

    elif selector_method in ['SR']:
        # Ref.1 3.1.4 SR from PLS
        # Selective - ratio
        # https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/pdf/10.1002/cem.1289
        def sr(model):
            t = model.x_scores_  # t = X * w
            w = model.x_weights_ # w = X.T * y / |X.T y| TP component vector
            p_TP = np.dot(self.X.T, t) / np.dot(t.T, t) 
            q = model.y_loadings_
            p, h = w.shape
            srs = np.zeros((p,))
            for i in range(p):
                v_exp = 1
                v_res = 1
                srs[i] = v_exp / v_res
            return srs

        scores = sr(estimator)

    elif selector_method in ['sMC', 'SMC']:
        # Ref.1 3.1.5 Significance multivariate correlation with PLS
        if threshold is None:
            raise ValueError('PLS-VIP needs the threshold as hyperparam')
            
        else:
            threshold = threshold

    elif selector_method in ['mRMR']:
        # Ref.1 3.1.5 Minimum redundancy maximum relevance in PLS
        if threshold is None:
            raise ValueError('PLS-VIP needs the threshold as hyperparam')
            
        else:
            threshold = threshold

    elif selector_method in ['GA-PLS','GAPLS','GA_PLS']:
        # Ref.1 3.2.1 Genetic algorithm combined with PLS regression
        if threshold is None:
            raise ValueError('PLS-VIP needs the threshold as hyperparam')
            
        else:
            threshold = threshold


    elif selector_method in ['SPA-PLS','PLS-SPA','SPAPLS','PLSSPA']:
        # Ref.1 3.2.2 Uninformative variable elimination in PLS
        if threshold is None:
            raise ValueError('PLS-VIP needs the threshold as hyperparam')
            
        else:
            threshold = threshold

    elif selector_method in ['SPA-PLS','PLS-SPA','SPA_PLS','PLS_SPA','SPAPLS','PLSSPA']:
        # Ref.1 3.2.3 Sub-window permutation analysis coupled with PLS
        if threshold is None:
            raise ValueError('PLS-VIP needs the threshold as hyperparam')
            
        else:
            threshold = threshold

    elif selector_method in ['PLS-IPW','IPW-PLS','PLSIPW','IPWPLS','PLS_IPW','IPW_PLS']:
        # Ref.1 3.2.4 Iterative predictor weighting PLS
        if threshold is None:
            raise ValueError('PLS-VIP needs the threshold as hyperparam')
            
        else:
            threshold = threshold

    elif selector_method in ['Backward']:
        # Ref.1 3.2.5 Backward variable elimination PLS
        if threshold is None:
            raise ValueError('PLS-VIP needs the threshold as hyperparam')
            
        else:
            threshold = threshold

    elif selector_method in ['REP']:
        # Ref.1 3.2.6 Regularized elimination procedure in PLS
        if threshold is None:
            raise ValueError('PLS-VIP needs the threshold as hyperparam')
            
        else:
            threshold = threshold

    elif selector_method in ['T2', 'Hotelling', 'Hotelling-T2', 'Hotelling_T2']:
        x_loadings_ = getattr(estimator, "x_loadings_", None)
        if x_loadings_ is not None:
            x_loadings_T = x_loadings_.T
            p = len(estimator.coef_)


            T2s = []
            for w_index, each_x_loadings_ in enumerate(x_loadings_):  
                each_x_loadings_ = np.array(each_x_loadings_).reshape(-1,1)
                each_x_loadings_mean = np.array(each_x_loadings_).reshape(-1,1).mean()
                
                x_loadings_ = np.array(x_loadings_)
                x_loadings_mean = x_loadings_.mean()

                x_x = np.dot(x_loadings_.T, x_loadings_)
                Sw_inv = np.linalg.inv(x_x)
        
                T2_ = p*(each_x_loadings_mean - x_loadings_mean)*Sw_inv*(each_x_loadings_mean - x_loadings_mean)    
                T2 = np.sum(T2_,axis=1)[0]
                T2s.append(T2)
            
            scores = T2s

        elif x_loadings_ is None:
            raise ValueError(
                "The underlying estimator %s has no `x_loadings_` attribute."
                " Either pass a fitted estimator to SelectFromModel or call fit before calling transform."
                % estimator.__class__.__name__)


    elif selector_method in ['ST', 'STPLS']:
        # Ref.1 3.3.1 ST PLS
        threshold = threshold

    elif selector_method in ['TRUNC', 'STPLS']:
        # Ref.1 3.3.2 Distribution-based truncation for variable selection in PLS 
        threshold = threshold

    elif selector_method in ['PLS-WVC', 'PLS_WVC', 'PLSWVC']:
        # Ref.1 3.3.3 Weighted variable contribution in PLS 
        threshold = threshold

    elif selector_method in ['RFR']:
        importances = getattr(estimator, "feature_importances_", None)
        if importances is not None:
            scores = importances
        else:
            raise ValueError(
                "The underlying estimator %s has no  "
                "`feature_importances_` attribute. Either pass a fitted estimator"
                " to SelectFromModel or call fit before calling transform."
                % estimator.__class__.__name__)


    scores = np.array(scores)

    return scores


def _get_selected(scores, threshold, min_features, max_features):
    if max_features is not None:
        if max_features < min_features:
            print('max_features {} <  min_features {}'.format(max_features, min_features))
            print('set max_features as min_features.')
            time.sleep(0.5)
            max_features = min_features
        
    if max_features is not None:
        selected = np.zeros_like(scores, dtype=bool)
        candidate_indices = \
            np.argsort(-scores, kind='mergesort')[:max_features]

        selected[candidate_indices] = True

    else:
        selected = np.ones_like(scores, dtype=bool)
    index = scores < threshold 
    selected[index] = False

    if min_features is not None:
        candidate_indices = \
            np.argsort(-scores, kind='mergesort')[:min_features]
        selected[candidate_indices] = True
    
    return selected


class SelectByModel(MetaEstimatorMixin, SelectorMixin, BaseEstimator):
    """Meta-transformer for selecting features based on importance weights.

    .. versionadded:: 0.17

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
        This can be both a fitted (if ``prefit`` is set to True)
        or a non-fitted estimator. The estimator must have either a
        ``feature_importances_`` or ``coef_`` attribute after fitting.

    threshold : string, float, optional default None
        The threshold value to use for feature selection. Features whose
        importance is greater or equal are kept while the others are
        discarded. If "median" (resp. "mean"), then the ``threshold`` value is
        the median (resp. the mean) of the feature importances. A scaling
        factor (e.g., "1.25*mean") may also be used. If None and if the
        estimator has a parameter penalty set to l1, either explicitly
        or implicitly (e.g, Lasso), the threshold used is 1e-5.
        Otherwise, "mean" is used by default.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. If the
        estimator is a classifier or if ``y`` is neither binary nor multiclass,
        :class:`sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value of None changed from 3-fold to 5-fold.

    prefit : bool, default False
        Whether a prefit model is expected to be passed into the constructor
        directly or not. If True, ``transform`` must be called directly
        and SelectFromModel cannot be used with ``cross_val_score``,
        ``GridSearchCV`` and similar utilities that clone the estimator.
        Otherwise train the model using ``fit`` and then ``transform`` to do
        feature selection.

    norm_order : non-zero int, inf, -inf, default 1
        Order of the norm used to filter the vectors of coefficients below
        ``threshold`` in the case where the ``coef_`` attribute of the
        estimator is of dimension 2.

    max_features : int or None, optional
        The maximum number of features selected scoring above ``threshold``.
        To disable ``threshold`` and only select based on ``max_features``,
        set ``threshold=-np.inf``.

        .. versionadded:: 0.20

    Attributes
    ----------
    estimator_ : an estimator
        The base estimator from which the transformer is built.
        This is stored only when a non-fitted estimator is passed to the
        ``SelectFromModel``, i.e when prefit is False.

    threshold_ : float
        The threshold value used for feature selection.

    Notes
    -----
    Allows NaN/Inf in the input if the underlying estimator does as well.

    Examples
    --------
    >>> from sklearn.feature_selection import SelectFromModel
    >>> from sklearn.linear_model import LogisticRegression
    >>> X = [[ 0.87, -1.34,  0.31 ],
    ...      [-2.79, -0.02, -0.85 ],
    ...      [-1.34, -0.48, -2.55 ],
    ...      [ 1.92,  1.48,  0.65 ]]
    >>> y = [0, 1, 0, 1]
    >>> selector = SelectFromModel(estimator=LogisticRegression()).fit(X, y)
    >>> selector.estimator_.coef_
    array([[-0.3252302 ,  0.83462377,  0.49750423]])
    >>> selector.threshold_
    0.55245...
    >>> selector.get_support()
    array([False,  True, False])
    >>> selector.transform(X)
    array([[-1.34],
           [-0.02],
           [-0.48],
           [ 1.48]])
    """
    def __init__(self, estimator, estimator_tuning=None, selector_tuning=None,
                 selector_method=None, tuning_method='optuna', threshold=None, percentile=None, magnification=1,  prefit=False,
                 norm_order=1, max_features=None, min_features=2, scores=None, cv=5, mask=None, recalculate=True):

        self.estimator = estimator
        self.estimator_tuning = estimator_tuning
        self.selector_tuning = selector_tuning
        self.selector_method = selector_method
        self.tuning_method = tuning_method
        self.threshold = threshold
        self.magnification = magnification
        self.percentile = percentile
        self.prefit = prefit
        self.norm_order = norm_order
        self.max_features = max_features
        self.min_features = min_features
        self.scores= scores
        self.cv = cv
        self.mask = mask
        self.recalculate = recalculate

        if self.estimator.__class__.__name__ == "PLSRegression":
            self.min_features = estimator.n_components

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
            
        scores = _get_scores(self.estimator_, self.selector_method, self.norm_order)

        if isinstance(self.threshold, str):
            if "*" in self.threshold:
                magnification, reference = self.threshold.split("*")
                self.magnification = float(magnification.strip())
                self.threshold = reference.strip()


        threshold = _calculate_threshold(self.estimator_, self.X, self.y,
                                         selector_method= self.selector_method,
                                         scores=scores, 
                                         threshold=self.threshold,
                                         magnification=self.magnification,
                                         percentile=self.percentile)
        self.scores = scores
        self.threshold = threshold
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
