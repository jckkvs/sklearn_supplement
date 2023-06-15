"""
The :mod:`sklearn.model_selection._search` includes utilities to fine-tune the
parameters of an estimator.
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>,
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Andreas Mueller <amueller@ais.uni-bonn.de>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Raghav RV <rvraghav93@gmail.com>
# License: BSD 3 clause
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence, Iterable
from functools import partial, reduce
from itertools import product
import numbers
import operator
from tabnanny import verbose
import time
import warnings

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.base import is_classifier
from sklearn.model_selection._split import check_cv


from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection._validation import _score


from sklearn.utils.validation import check_is_fitted
from sklearn.metrics._scorer import _check_multimetric_scoring, make_scorer

from sklearn.metrics import (
    r2_score,
    median_absolute_error,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    mean_poisson_deviance,
    mean_gamma_deviance,
    accuracy_score,
    top_k_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    log_loss,
    balanced_accuracy_score,
    explained_variance_score,
    brier_score_loss,
    jaccard_score,
    mean_absolute_percentage_error,
)


SCORES = dict(
    r2=r2_score,
    accuracy=accuracy_score,
    roc_auc=roc_auc_score,
    # Cluster metrics that use supervised evaluation
)

from sklearn.metrics import check_scoring

from sklearn_expansion.model_selection._validation import cross_val_score, cross_val_predict


import optuna
from optuna.logging import set_verbosity

from ..metrics._scorer import k3nerror
k3n_error = make_scorer(k3nerror, greater_is_better=False)

__all__ = ['GridSearchCV', 'OptunaSearchCV', 'ParameterGrid', 'fit_grid_point',
           'ParameterSampler', 'RandomizedSearchCV']


class OptunaSearchCV(BaseSearchCV):
    _required_parameters = ["estimator", "param_range"]

    def __init__(
        self,
        estimator,
        param_range,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=30,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
        timeout=None,
        n_trials=None
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

        self.estimator = estimator
        self.param_range = param_range

        self.cv=check_cv(cv)

        self.scoring=scoring
        self.n_jobs=n_jobs
        self.refit=refit
        self.verbose=verbose
        self.pre_dispatch=pre_dispatch
        self.error_score=error_score
        self.return_train_score=return_train_score
        self.timeout=timeout
        self.n_trials=n_trials

        #print('optimize cv' , self.cv)


    def fit(self, X, y,*, groups=None, **fit_params):
        """Search all candidates in param_grid"""
        def objective_estimator(trial):
            def get_hyperparams(need_tuning_hyper_params):
                # tuningするhyperparameterをparamに格納
                suggest_dict = {'log'       :trial.suggest_loguniform,
                            'loguniform':trial.suggest_loguniform,
                            'uniform'   :trial.suggest_uniform,
                            'int'       :trial.suggest_int,
                            'choice'    :trial.suggest_categorical,
                            'category'  :trial.suggest_categorical,
                            'categorical':trial.suggest_categorical}

                param = {}
                for each_hyperparam_name, each_hyperparam_value in need_tuning_hyper_params.items():                
                    suggest_type = each_hyperparam_value['suggest_type']
                    if suggest_type in suggest_dict.keys():  
                        param_name       = each_hyperparam_name
                        trial_suggest    = suggest_dict[suggest_type]                  
                        search_range     = each_hyperparam_value['range']
                        param[param_name]= trial_suggest(param_name, *search_range)

                    else:
                        print('suggest type {} is not defined'.format(suggest_type))
                        print('you should choose from {}'.format(suggest_dict.keys()))
                        time.sleep(3)

                return param

            estimator_need_tuning_hyper_params = get_hyperparams(need_tuning_hyper_params)

            # V0.0.4
            for key, item in estimator_need_tuning_hyper_params.items():
                setattr(estimator, key, item)

            # removed in v.0.0.3
            #estimator.set_params(**estimator_need_tuning_hyper_params)

            if callable(self.scoring):
                scorers = self.scoring
            elif isinstance(self.scoring, str):
                scorers = SCORERS[self.scoring]
            elif self.scoring is None:
                if is_classifier(self.estimator) == False:
                    scorers = r2_score
                else:
                    scorers = accuracy_score
            else:
                scorers = _check_multimetric_scoring(self.estimator, self.scoring)
                self._check_refit_for_multimetric(scorers)
                refit_metric = self.refit

            y_pred_cv = cross_val_predict(estimator, X, y, groups=groups, cv=self.cv) #, scoring = scorers))
            scores = scorers(y, y_pred_cv)
            score = scores.mean()

            return score

        estimator = self.estimator
        need_tuning_hyper_params = self.param_range

        if groups is None:
            pass

        # PLSにおいてn_componentsの探索範囲をXのrank数以内に修正
        if 'n_components' in need_tuning_hyper_params.keys():
            x_rank = np.linalg.matrix_rank(X)

            # choiceの場合
            if need_tuning_hyper_params['n_components']['suggest_type'] == 'choice':           
                need_tuning_hyper_params['n_components']['range'] = sorted(list(set([min(i,x_rank) for i in need_tuning_hyper_params['n_components']['range']])))
            
            # intの場合 param_range = {'n_components' : {'suggest_type':'int', 'range':[1,30]}}
            elif need_tuning_hyper_params['n_components']['suggest_type'] == 'int':         
                need_tuning_hyper_params['n_components']['range'][1] = min(need_tuning_hyper_params['n_components']['range'][1], x_rank)

        n_need_tuning_hyper_params = len(need_tuning_hyper_params)

        if n_need_tuning_hyper_params != 0:
            # hyperparameter の数に応じてn_trialsを変更 最大でも15回
            if self.n_trials is None:
                n_trials = min(n_need_tuning_hyper_params**2 *5 +20, 100)
            else:
                n_trials = self.n_trials
            if self.timeout is None:
                # hyperparameter の数に応じてn_trialsを変更 最大でも600秒
                timeout = 600
            else:
                timeout = self.timeout


            set_verbosity(self.verbose)
            study = optuna.create_study(direction='maximize')
            study.optimize(objective_estimator, n_trials=n_trials, timeout=timeout)
            #estimator_tuned_params = study.best_params

            self.best_params_ = study.best_params
            best_estimator_ = estimator.set_params(**self.best_params_)
            self.best_estimator_ = best_estimator_
            
        else:
            print(' no tuned')
            #estimator_tuned_params = {}
            self.best_params_ = {}
            self.best_estimator_ =  self.estimator

        if self.refit:
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
        
        return self

    def predict(self, X):
        check_is_fitted(self)        
        print(self.best_estimator_)
        return self.best_estimator_.predict(X)



class OptunaSearchClustering(TransformerMixin, BaseSearchCV):

    _required_parameters = ["estimator", "param_range"]

    def __init__(
        self,
        estimator,
        param_range,
        *,
        scoring=k3nerror,
        n_jobs=None,
        refit=True,
        verbose=30,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

        self.param_range = param_range

    def fit(self, X, **fit_params):
        """Search all candidates in param_grid"""
        def objective_estimator(trial):
            def get_hyperparams(need_tuning_hyper_params):
                # tuningするhyperparameterをparamに格納
                suggest_dict = {'log'       :trial.suggest_loguniform,
                            'loguniform':trial.suggest_loguniform,
                            'uniform'   :trial.suggest_uniform,
                            'int'       :trial.suggest_int,
                            'choice'    :trial.suggest_categorical,
                            'category'  :trial.suggest_categorical,
                            'categorical':trial.suggest_categorical}

                param = {}
                for each_hyperparam_name, each_hyperparam_value in need_tuning_hyper_params.items():                
                    suggest_type = each_hyperparam_value['suggest_type']
                    if suggest_type in suggest_dict.keys():  
                        param_name       = each_hyperparam_name
                        trial_suggest    = suggest_dict[suggest_type]                  
                        search_range     = each_hyperparam_value['range']
                        param[param_name]= trial_suggest(param_name, *search_range)

                    else:
                        print('suggest type {} is not defined'.format(suggest_type))
                        print('you should choose from {}'.format(suggest_dict.keys()))
                        time.sleep(3)

                return param

            estimator_need_tuning_hyper_params = get_hyperparams(need_tuning_hyper_params)

            # V0.0.4
            for key, item in estimator_need_tuning_hyper_params.items():
                setattr(estimator, key, item)

            # removed in v.0.0.3
            #estimator.set_params(**estimator_need_tuning_hyper_params)

            if callable(self.scoring):
                scorers = self.scoring
            elif self.scoring is None or isinstance(self.scoring, str):
                #scorers = check_scoring(self.estimator, self.scoring)
                scorers = SCORES[self.scoring]
                #scorers = None
            else:
                scorers = _check_multimetric_scoring(self.estimator, self.scoring)
                self._check_refit_for_multimetric(scorers)
                refit_metric = self.refit

            if hasattr(self, 'transform') == True:                
                X_transformed = estimator.fit_transform(X)
                #scores = scorers(X, X_transformed)
                scores = scorers(X, X_transformed)
                score = -scores.mean()
            else:
                score = 0

            return score

        estimator = self.estimator
        need_tuning_hyper_params = self.param_range
        n_need_tuning_hyper_params = len(need_tuning_hyper_params)

        if n_need_tuning_hyper_params != 0:
            # hyperparameter の数に応じてn_trialsを変更 最大でも15回
            n_trials = min(n_need_tuning_hyper_params**2 *5 +20, 100)

            # hyperparameter の数に応じてn_trialsを変更 最大でも600秒
            timeout = 600

            set_verbosity(self.verbose)
            study = optuna.create_study(direction='maximize')
            study.optimize(objective_estimator, n_trials=n_trials, timeout=timeout)
            estimator_tuned_params = study.best_params
            self.best_params_ = study.best_params
            print('self.best_params_')
            print(self.best_params_)
            best_estimator_ = estimator.set_params(**self.best_params_)
            self.best_estimator_ = best_estimator_
            
        else:
            print(f'{self.estimator.__class__.__name__} have no tuning parameters')
            estimator_tuned_params = {}
            self.best_params_ = {}
            self.best_estimator_ =  self.estimator

        if self.refit:
            self.best_estimator_.fit(X, **fit_params)
        
        return self

    def predict(self, X):
        check_is_fitted(self)        
        return self.best_estimator_.predict(X)

    def transform(self, X):
        if hasattr(self.best_estimator_, 'transform') == True:
            return self.best_estimator_.transform(X)
        elif hasattr(self.best_estimator_, 'fit_transform'):
            # TSNE etc. cannot apply the 'transform' only. Always they need fit_transform.
            return self.best_estimator_.fit_transform(X)
        else:
            raise AttributeError('')


