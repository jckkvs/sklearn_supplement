import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy import special
#from minepy import MINE

def calc_corr(df, method='pearson'):
    df = pd.DataFrame(df)

    corr_df = df.corr()
    n = len(df)

    if pd.__version__ <= "1.2":
        print('change method to pearson')
        method = 'pearson'

    if method=='pearson':
        pass
    elif method=='mvue':
        def mvue_corr(x,n):
            mvue_r = x * special.hyp2f1(0.5, 0.5, (n-1)/2, 1-x**2)
            return mvue_r

        #corr_df = corr_df.applymap(mvue_corr, {'n':n})
        corr_df = corr_df.applymap(lambda x:  x * special.hyp2f1(0.5, 0.5, (n-1)/2, 1-x**2), na_action='ignore')
        # https://hippocampus-garden.com/stats_correlation_bias/
        # https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-29/issue-1/Unbiased-Estimation-of-Certain-Correlation-Coefficients/10.1214/aoms/1177706717.full

    elif method=='dj':
        def dj_corr(x,n):
            dj_r = x * (1 + (1 - x**2) / (2 * (n-3)))
            return dj_r
        # https://hippocampus-garden.com/stats_correlation_bias/
        #corr_df = corr_df.applymap(dj_corr, {'n':n})
        corr_df = corr_df.applymap(lambda x:  x * (1 + (1 - x**2) / (2 * (n-3))), na_action='ignore')
    else:
        pass

    return corr_df
