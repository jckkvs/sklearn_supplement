import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def calculate_threshold(X, *, n_neighbors=5, percentile=0.997, metric='euclidean'):
    """calculate_threshold by nearest neighbors

    Parameters
    ----------
    X : pd.DataFrame
        X should be autoscaled.
    n_neighbors : int
        Number of neighbors to use by default for :meth:`kneighbors` queries.        
    percentile : float
        The value of the percentile that determines the sample to be the value of the threshold.
        The top X% points of the samples that are far from the average
         distance from other samples are specified. 
    metric : str or callable, default='euclidean'
        The distance metric to use for the tree.  The default metric is
        euclidean metric. For a list of available metrics, see the documentation of
        :class:`~scipy.spatial.distance.cdist`.

    Returns
    ----------
    threshold : float
        The threshold of applicability domain.

    """

    X = np.array(X)
    n_samples = X.shape[0]
    distance_all = cdist(X, X, metric=metric)
    distance_all = np.sort(distance_all, axis=1)
    distance_all = distance_all[:, 1:(n_neighbors+1)]

    knndistance = np.mean(distance_all, axis=1)
    knndistance = np.sort(knndistance)
    # 他のサンプルとの平均距離が遠い上位X%を閾値とする
    threshold = knndistance[round(n_samples * percentile) - 1]

    return threshold

def check_within_ad(X_train, X_test, threshold, *, n_neighbors=3, metric='euclidean', return_distance=False):
    """ Calculate if X_test is in the applicability domain of X_train

    Parameters
    ----------
    X_train : pd.DataFrame
        X should be autoscaled.
    X_test : pd.DataFrame
        X should be autoscaled.
    threshold : float
        The threshold of applicability domain.       
    n_neighbors : int, optional
        Number of neighbors to use by default for 3
    metric : str or callable, default='euclidean'
        The distance metric to use for the tree.  The default metric is
        euclidean metric. For a list of available metrics, see the documentation of
        :class:`~scipy.spatial.distance.cdist`.
    return_distance : boolean
        
    Returns
    -------
    within_AD : numpy.arrays of boolean
    distance_mean : numpy.arrays
        Optional. The average distance of X_test to X_train within kNN

    """

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # 訓練データと予測用データの距離計算　cdist
    distance_all = cdist(X_train, X_test, metric=metric)
    # 距離を昇順でソート
    distance_sorted = np.sort(distance_all, axis=0)
    # kNearest Neighbors までの平均を取る
    distance_knn = distance_sorted[0:n_neighbors]
    average_distance = np.mean(distance_knn, axis=0)

    distance_df = pd.DataFrame(average_distance)
    within_ad = np.array(distance_df.iloc[:,0].apply(lambda x : True if  x<= threshold else False))

    if return_distance == False:
        return within_ad
    else:
        return within_ad, average_distance