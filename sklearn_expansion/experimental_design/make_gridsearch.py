
import math
import random
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def make_candidates(candidates):
    candidate_list = []
    n_explanatory = len(candidates)
    for i in range(n_explanatory):
        if (isinstance(candidates[i], list)):
            each_candidates = np.array(candidates[i])
            candidate_list.append(each_candidates)        
    candidate_np = np.array(candidate_list)
    return candidate_np

def make_candidates(candidates):
    candidate_list = []

    n_explanatory = len(candidates)
    
    for i in range(n_explanatory):
        if (isinstance(candidates[i], str)):
            each_candidates = candidates[i].split(",")
            each_candidates = np.array([float(i) for i in each_candidates])
            candidate_list.append(each_candidates)
        elif (isinstance(candidates[i], float)) or (isinstance(candidates[i], int)):
            each_candidates = np.array([candidates[i]])
            candidate_list.append(each_candidates)        
        else:            
            if split_num[i] <= 1:        
                # 分割せずに、最頻値を採用する
                candidate_list.append(np.array([mode_[i]])) 
            else:
                candidate_list.append(np.linspace(start = min_[i], stop = max_[i], num = split_num[i])) #分割を実行
    
    candidate_np = np.array(candidate_list)
    return candidate_np

def make_candidate(maxs, mins, modes, split_num,):
    return candidates    
