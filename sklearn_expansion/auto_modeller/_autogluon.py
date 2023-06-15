# Author: Yuki Horie <Yuki.Horie@mitsuichemicals.com>


import numpy as np
import pandas as pd

import autogluon as ag
from autogluon.tabular  import TabularPredictor

from sklearn.base import BaseEstimator, TransformerMixin

class AutoGluonPredictor(TabularPredictor):
    def __init__(self,
                 label='',
                 problem_type=None,
                 eval_metric=None,
                 path=None,
                 verbosity=0,
                 sample_weight=None,
                 weight_evaluation=False,
                 groups=None,
                 presets='best_quality'):
        super().__init__(label=label,
                         problem_type=problem_type,
                         eval_metric=eval_metric,
                         path=path,
                         verbosity=verbosity,
                         sample_weight=sample_weight,
                         weight_evaluation=weight_evaluation,
                         groups=groups)
        
        self.presets = presets

    def fit(self, X, y=None, groups=None, sample_weight=None, check_input=True):
        X_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y)
        X_names = X_df.columns
        y_names = y_df.columns

        if len(X_names) != len(set(X_names)):
            X_names = [f'X{i}' for i in range(len(X_names))]
        else:
            if len(list(set(X_names) & set(y_names))) >= 1 :
                X_names = [f'X{i}' for i in range(len(X_names))]
                y_names = ['y0']

        X_df.columns = X_names
        y_df.columns = y_names

        y_name = y_names[0]

        self.X_names = X_names
        self.y_names = y_names

        self.__init__(label=y_name)
        train_data = pd.concat([X_df, y_df], axis=1)
        super().fit(train_data, presets=self.presets)    
        return self

    def predict(self, X, y=None, groups=None,sample_weight=None, check_input=True):
        X_df = pd.DataFrame(X)
        X_df.columns = self.X_names

        return super().predict(X_df)


    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """
        Parameters
        ----------
        deep : Ignored. (for compatibility with sklearn)
        Returns
        ----------
        self : returns an dictionary of parameters.
        """

        params = {}
        return params        