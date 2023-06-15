import numpy as np
from numpy import matlib
import pandas as pd
import copy

def kennard_stone(X, n_candidates=10,  *, metric='euclidean'):
        
    """create experimental design candidates.
    Parameters
    ----------
    X : np.array
        X should be autoscaled.
    n_candidates : int
    metric : str or callable, default='euclidean'
        Ignored.
        The distance metric to use for the tree.  The default metric is
        euclidean metric. For a list of available metrics, see the documentation of
        :class:`~scipy.spatial.distance.cdist`.
    Returns
    ----------
    canididates : np.array

    Notes
    ----------
    metrics parameteres : Ignored. 
    Please extend the function to the jaccard index and so on. 

    """

    AllX = copy.deepcopy(X)

    DistanceToAverage = ( (X - np.matlib.repmat(X.mean(axis=0), X.shape[0], 1) )**2 ).sum(axis=1)
    MaxDistanceSampleNumber = np.where( DistanceToAverage == np.max(DistanceToAverage) )
    MaxDistanceSampleNumber = MaxDistanceSampleNumber[0][0]
    SelectedSampleNumbers = list()
    SelectedSampleNumbers.append(MaxDistanceSampleNumber)
    RemainingSampleNumbers = np.arange( 0, X.shape[0], 1)
    X = np.delete( X, SelectedSampleNumbers, 0)
    RemainingSampleNumbers = np.delete( RemainingSampleNumbers, SelectedSampleNumbers, 0)
    IterationNumbers = np.arange( 1, n_candidates, 1)
    for _ in IterationNumbers:
        SelectedSamples = AllX[SelectedSampleNumbers,:]
        # 2. For each sample that has not been selected yet, the distance from the already-selected data is calculated and the minimum distance is saved
        MinDistanceToSelectedSamples = list()
        MinDistanceCalcNumbers = np.arange( 0, X.shape[0], 1)
        for MinDistanceCalcNumber in MinDistanceCalcNumbers:
            DistanceToSelectedSamples = ( (SelectedSamples - np.matlib.repmat(X[MinDistanceCalcNumber,:], SelectedSamples.shape[0], 1) )**2 ).sum(axis=1)
            MinDistanceToSelectedSamples.append( np.min(DistanceToSelectedSamples) )
            # 3. Sample whose minimum distance is the largest is newly selected    
        MaxDistanceSampleNumber = np.where( MinDistanceToSelectedSamples == np.max(MinDistanceToSelectedSamples) )
        MaxDistanceSampleNumber = MaxDistanceSampleNumber[0][0]
        SelectedSampleNumbers.append(RemainingSampleNumbers[MaxDistanceSampleNumber])
        X = np.delete( X, MaxDistanceSampleNumber, 0)
        RemainingSampleNumbers = np.delete( RemainingSampleNumbers, MaxDistanceSampleNumber, 0)

    candidates = X[SelectedSampleNumbers,:]

    return candidates