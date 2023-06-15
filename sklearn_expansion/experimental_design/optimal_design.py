
import math
import random
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# https://oceanone.hatenablog.com/entry/2021/01/28/025000

def calc_rank(X):
    return np.linalg.matrix_rank(np.dot(X.T, X))

def covert_low_rank_matrix(X, rank=2):
    pca = PCA(n_components=rank)
    X_low_rank_matrix = pca.fit_transform(X)
    return X_low_rank_matrix

def calc_A_value(X):    
    det = np.linalg.det(np.dot(X.T, X)) 
    if det >= 1E-6:
        return np.trace(np.linalg.inv(np.dot(X.T, X)))
    else:
        return np.inf

def A_optimal_design(X_designs, n_jobs=None, low_rank=False, verbose=0):
    '''
    Parameters
    -------
    X_designs : list of numpy array
        Each X should be standarized.
    Returns
    -------
    X : numpy array
        Selected X by optimal_design
    '''
    
    # traces = np.array([calc_A_value(X) for X in X_designs])
    traces = np.array([joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(calc_A_value)(X) for X in X_designs)])[0]
    
    min_trace = min(traces)
    min_X = X_designs[np.argmin(traces)] 

    if min_trace <= 1E-6:
        raise ValueError("Minimum trace too low. The experimental design may be collinear, or the number of explanatory variables may exceed the number of samples")

    return min_X, min_trace

def calc_D_value(X):
    return np.linalg.det(np.dot(X.T, X))

def D_optimal_design(X_designs, *, n_jobs=None, low_rank=False, verbose=0):
    '''
    Parameters
    -------
    X : list of numpy array
        Each X should be standarized.
    Returns
    -------
    X : numpy array
        Selected X by optimal_design
    '''

    determinants = np.array([joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(calc_D_value)(X) for X in X_designs)])[0]

    max_determinant = max(determinants)
    if max_determinant <= 1E-6:
        raise ValueError("Maximum determinant too low. The experimental design may be collinear, or the number of explanatory variables may exceed the number of samples")

    max_X = X_designs[np.argmax(determinants)] 

    return max_X, max_determinant

def calc_E_value(X):
    return min(np.linalg.eigvals(np.dot(X.T, X)))

def E_optimal_design(X_designs, n_jobs=None, low_rank=False, verbose=0):
    '''
    Parameters
    -------
    X_designs : list of numpy array
        Each X should be standarized.
    Returns
    -------
    X : numpy array
        Selected X by optimal_design
    '''
    
    # eigens = np.array([calc_E_value(X) for X in X_designs])
    eigens = np.array([joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(calc_E_value)(X) for X in X_designs)])[0]
    max_eigen = max(eigens)
    if max_eigen <= 1E-6:
        raise ValueError("Maximum eigen value too low. The experimental design may be collinear, or the number of explanatory variables may exceed the number of samples")

    max_X = X_designs[np.argmax(eigens)] 
    return max_X, max_eigen

def calc_E2_value(X):
    eigvals = np.linalg.eigvals(np.dot(X.T, X))
    eigvals_ = np.where(eigvals < 1E-6, np.nan, eigvals)
    return np.nanmin(eigvals_), np.isnan(eigvals_).sum()

def calc_E2_value_(X, n_replace):
    eigvals = np.linalg.eigvals(np.dot(X.T, X))
    eigvals[-int(n_replace):] = np.nan
    return np.nanmin(eigvals)

def E2_optimal_design(X_designs, n_jobs=None, low_rank=False, verbose=0):
    '''
    Parameters
    -------
    X_designs : list of numpy array
        Each X should be standarized.
    Returns
    -------
    X : numpy array
        Selected X by optimal_design
    '''
    
    results = np.array([joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(calc_E2_value)(X) for X in X_designs)])[0]
    min_eigens, n_nan = zip(*results)
    unique_n_nan = np.unique(n_nan)
    min_n_nan = int(np.real(min(unique_n_nan)))

    if len(unique_n_nan) >= 2 and min_n_nan != 0:
        min_eigens = np.array([joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(calc_E2_value_)(X, min_n_nan) for X in X_designs)])[0]
    else:
        pass
    
    min_eigens = np.real_if_close(min_eigens)
    max_min_eigen = max(min_eigens)
    max_X = X_designs[np.argmax(min_eigens)] 
    return max_X, max_min_eigen

def calc_I_value(X_design, X_design_space):

    # n = len(X_design)    
    m = np.dot(X_design.T, X_design) # / n
    if np.linalg.det(m) == 0 :
        return np.inf

    m_inverse = np.linalg.inv(m)
    diag =  np.diag(np.dot(np.dot(X_design_space, m_inverse), X_design_space.T))

    value = np.sum(diag)    
    return value

def I_optimal_design(X_designs, X_design_space, n_jobs=None, low_rank=False, verbose=0):
    '''
    Parameters
    -------
    X_designs : list of numpy array
        Each array is design candidates.
        Each X should be standarized by X_design_space
    X_design_space : list of numpy array
        All design space over X.        
    Returns
    -------
    X : numpy array
        Selected X by optimal_design
    '''
    X_design_space = np.array(X_design_space)

    mean_diags = np.array([joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(calc_I_value)(X, X_design_space) for X in X_designs)])[0]

    min_mean_diag = min(mean_diags)
    if min_mean_diag >= 1E+6:
        raise ValueError("Minimum I-Value too high. The experimental design may be collinear, or the number of explanatory variables may exceed the number of samples")

    min_X = X_designs[np.argmin(mean_diags)] 


    return min_X, min_mean_diag

def calc_G_value(X_design, X_design_space):

    m = np.dot(X_design.T, X_design)
    if np.linalg.det(m) == 0 :
        return np.inf

    m_inverse = np.linalg.inv(m)
    diag =  np.diag(np.dot(np.dot(X_design_space, m_inverse), X_design_space.T))
    value = np.max(diag)

    return value

def G_optimal_design(X_designs, X_design_space, n_jobs=None, low_rank=False, verbose=0):
    '''
    Parameters
    -------
    X_designs : list of numpy array
        Each array is design candidates.
        Each X should be standarized by X_design_space
    Returns
    -------
    X : numpy array
        Selected X by optimal_design
    Notes
    -------    
    https://cs.nju.edu.cn/zlj/pdf/AAAI-2010-Chen.pdf
    '''
    X_design_space = np.array(X_design_space)
    max_diags = np.array([joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(calc_G_value)(X, X_design_space) for X in X_designs)])[0]
    min_max_diag = min(max_diags)
    min_X = X_designs[np.argmin(max_diags)] 

    return min_X, min_max_diag

def create_optimal_design(X_design_space, optimal_type='D', n_experiments=10,  *, X_fixed_design=None, n_searches=10000, random_state=None, n_jobs=None, low_rank=True, verbose=0):
        
    """create experimental design candidates.

    Parameters
    ----------
    X_design_space : list (TODO iterator)
        X_design_space should be autoscaled.
    optimal_type : str, optional
        `D`:D-optimal design,  `I`:I-optimal design, 
    n_experiments : int, optional
        A number of designs in the experimental design
    X_fixed_design : list, optional
        An experimental candidate or a candidate that has already been experimented. 
        Be sure to use it when calculating the valuation value.
    n_searches : int, optional
        A number of times to compute the best fit criterion for random candidates
    random_state : int, RandomState instance or None, default=None, optional
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    low_rank : boolean, optional

    verbose : int, optional
        The verbosity level: if non zero, progress messages are
        printed. Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.

    Returns
    ----------
    candidates : np.array
        The candidates which select by optimal_type.

    Examples
    --------
    >>> import numpy as np
    >>> X_1 = np.arange(-2, 2.4 ,0.4)
    >>> X_2 = np.arange(-2, 2.4 ,0.4)
    >>> X_fixed_design = [[0,0], [1.0, 1.0],[-1.0, -1.0]]
    >>> X_design_space = list(itertools.product(X_1, X_2))
    >>> create_optimal_design(X_design_space, optimal_type='D', n_experiments=10, n_searches=10000, random_state=None)
    
    """


    random.seed(random_state)
    sscaler  = StandardScaler()
    X_design_space = sscaler.fit_transform(X_design_space)

    if len(X_design_space) >= 10000000:
        warnings.warn("X_design_space are too big. It could be slower. You should the number of candidate space.")
        time.sleep(3.5)

    if low_rank == True:
        rank = calc_rank(X_design_space)
        pca = PCA(n_components=rank)
        X_design_space_ = pca.fit_transform(X_design_space)
        X_designs = [np.array(random.sample(list(X_design_space_), n_experiments)) for i in range(n_searches)]

        if X_fixed_design is not None:
            X_fixed_design = sscaler.transform(X_fixed_design)
            X_fixed_design_ = pca.fit_transform(X_fixed_design)
            X_designs = [np.concatenate([i, X_fixed_design_], axis=0) for i in X_designs]

    else:
        X_design_space_ = X_design_space
        X_designs = [np.array(random.sample(list(X_design_space_), n_experiments)) for i in range(n_searches)]

        if X_fixed_design is not None:
            X_fixed_design = sscaler.transform(X_fixed_design)
            X_designs = [np.concatenate([i, X_fixed_design], axis=0) for i in X_designs]


    if optimal_type == "A":
        X_optimal_design_, value = A_optimal_design(X_designs, n_jobs=n_jobs, low_rank=low_rank, verbose=verbose)
    elif optimal_type == "D":
        X_optimal_design_, value = D_optimal_design(X_designs, n_jobs=n_jobs, low_rank=low_rank, verbose=verbose)
    elif optimal_type == "E":
        X_optimal_design_, value = E_optimal_design(X_designs, n_jobs=n_jobs, low_rank=low_rank, verbose=verbose)
    elif optimal_type == "E2":
        X_optimal_design_, value = E2_optimal_design(X_designs, n_jobs=n_jobs, low_rank=low_rank, verbose=verbose)
    elif optimal_type == "I":
        X_optimal_design_, value = I_optimal_design(X_designs, X_design_space_, low_rank=low_rank, n_jobs=n_jobs, verbose=verbose)
    elif optimal_type == "G":
        X_optimal_design_, value = G_optimal_design(X_designs, X_design_space_, low_rank=low_rank, n_jobs=n_jobs, verbose=verbose)
    elif optimal_type == "random":
        X_optimal_design_ = X_design_space_[0]
        value = np.nan
    else:
        raise ValueError(f'optimal type {optimal_type} is not supported.')

    print(f"{optimal_type} value is {value}")

    if low_rank == True:
        X_optimal_design = pca.inverse_transform(X_optimal_design_)
    else:
        X_optimal_design = X_optimal_design_


    X_optimal_design_raw = sscaler.inverse_transform(X_optimal_design)

    return X_optimal_design_raw