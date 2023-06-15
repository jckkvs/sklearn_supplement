# Author: Yuki Horie <Yuki.Horie@mitsuichemicals.com>


import numpy as np
import pandas as pd

from pytorch_tabnet.tab_model import TabNetRegressor as TNR
import torch

from sklearn.base import BaseEstimator, TransformerMixin

class TabNetRegressor(TNR):
    def __post_init__(self):
        super(TabNetRegressor, self).__post_init__()
        self._task = 'regression'
        self._default_loss = torch.nn.functional.mse_loss
        self._default_metric = 'mse'


    def fit(self, X, y=None, groups=None, sample_weight=None, check_input=True):
        print('tabnet fit')
        if type(X) is pd.core.frame.DataFrame:
            X_df = X
        elif type(X) is np.ndarray:
            X_df = pd.DataFrame(X)         
        elif type(X) is list:
            X_df = pd.DataFrame(X)         
        else:
            raise ValueError   

        if type(y) is pd.core.frame.DataFrame:
            y_df = y
        elif type(y) is np.ndarray:
            y_df = pd.DataFrame(y)         
        elif type(X) is list:
            y_df = pd.DataFrame(y)         
        else:
            raise ValueError   

        X_names = X_df.columns
        y_names = y_df.columns

        if len(X_names) != len(set(X_names)):
            print(f'X_names didnot match')
            X_names = [f'X{i}' for i in range(len(X_names))]
        else:
            if len(list(set(X_names) & set(y_names))) >= 1 :
                X_names = [f'X{i}' for i in range(len(X_names))]
                y_names = ['y0']

        X_df.columns = X_names
        y_df.columns = y_names

        y_name = y_names[0]


        print(X_names)
        print(y_names)

        print(X_df)
        print(y_df)

        self.X_names = X_names
        self.y_names = y_names

        super().fit(X_df, y_df)    
        return self

    def predict(self, X, y=None, groups=None,sample_weight=None, check_input=True):
        print('tabnet predict')
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