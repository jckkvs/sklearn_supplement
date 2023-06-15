import numpy as np 
import random
import joblib
from scipy import stats
import time
import sys
import pprint
import copy

"""ランダムサンプリングに用いる制約条件や生成関数の関数群とその使用例
ランダムサンプリングは「乱数を発生して制約条件を満たすサンプルを採用する」という手法も可能だが、
厳しい制約条件を課すと採用確率が低下するため、ランダムサンプリングに必要な時間が増加していく
*採用確率が0.01%ならランダムサンプリングに要する時間が10000倍となる

従って、
「制約条件を満たすまでランダムサンプリングを繰り返す」のではなく
「制約条件を満たす様な生成関数に基づきランダムサンプリングする」ことが望ましい。

ただ、生成関数で表現するのが難しい条件もあるため、制約条件・生成関数の両方に対応する必要がある
*例えば、組成1 × 組成3 が0.3以上、0.5以下　など

使用例:
説明変数が5個の場合,まずsample = np.zeros(5) ~ [0,0,0,0,0]を作成します
そのsampleに対して様々な制約条件、設定条件を適用することで所望のsampleを得る
例えば、0以上1未満の一様乱数を設定するgenerate_value_by_uniform関数(生成条件)を適用すると
sample = [0.24, 0.15, 0.63, 0.91, 0.45]を得ることが出来ます
次に、合計を1とするset_sum_to_1関数(生成条件)を適用すると
各要素を合計(0.24+0.15+0.63+0.91+0.45=2.38)で除することで
sample = [0.1008, 0.063, 0.264, 0.382, 0.189]を得ることが出来ます

「組成1 × 組成3 が0.3以上、0.5以下」という制約条件を満たしたい場合は
lambda sample: sample[1] * sample[3]>=0.3 and sample[1] * sample[3] <=0.5}　という関数を適用する

こうした関数群をrulesとしてまとめて、apply_rules_to_n_samples関数に渡すことで
所望の個数のランダムサンプルを得ることができます
random_samples = apply_rules_to_n_samples(sample, rules, n_samples=20)

apply関数
 apply_rules_to_sample : 1つのサンプルに対してrulesを適用する
 apply_rules_to_samples : 与えられたサンプル群に対してrulesを適用する
 apply_rules_to_n_samples : 1つのサンプルに対してapply_rules_to_samplesを複数回適用する

rule関数
 generate_value_by_uniform : 一様分布に基づき乱数を発生する
 generate_value_by_dirichlet : ディリクレ分布に基づき乱数を発生する
 generate_value_by_choice : 与えられた候補から各要素を選択する
 set_sum_to_1 : 指定したインデックス群の合計を1とする
 set_lower_bound : 下限値を設定する
 set_upper_bound : 上限値を設定する
 set_bounds : 下限と上限を設定する
 set_ratio_range : 指定したインデックスの比率を範囲を指定する
 set_ratio_range : 指定したインデックス群の比率を設定する
 set_exclusive : 排他的に要素を設定する

"""


def apply_rules_to_sample(sample, rules=None,*,repeat=False):
    """ apply_rules_to_sample
        1つのsampleに対して、rulesを適用する関数

    Parameters
    ----------
    sample: ndarray [N, 1]
        the array of sample

    rule: list of dictionary
        the rule of sampling
         do : Function to match the value of each element to the condition
         constraints : 
    repeat: boolean
        If repeat == True, repeat sample generation until samples pass constraints.
        Else if repeat == False, returns None if the sample does not pass the constraint

    Returns
    -------
    sample: np.array
        the array of sample following to rules

    Examples
    --------
    >>> sample = np.zeros(10)
    >>> rules = [
            {"do": lambda i: generate_value_by_uniform(i)},
            {"do": lambda i: i/sum(i)},
            {"constraints": lambda i: i[0]>=0.1 and i[0]<=0.5},
            {"constraints": lambda i: (i[0] + i[1])<0.6},
            {"constraints": lambda i: i[3]>=0.2},
            ]    
    >>> apply_rules_to_sample(sample, rules) 
        [0.1148768  0.09509002 0.01537105 0.21836462 0.30369149 0.01273166
         0.06836711 0.13059971 0.03994358 0.00096398]

    >>> apply_rules_to_n_samples(sample, n=10000, rules=rules) 

    """

    if rules is None:
        return sample
    
    new_sample = copy.deepcopy(sample)    
    pass_rules = False

    while pass_rules == False:
        pass_rules = True
        for rule in rules:
            if 'do' in rule.keys():
                # do 条件の場合はその関数をsampleに適用する
                func = rule["do"]
                new_sample = func(new_sample)
                
            elif 'constraints' in rule.keys():
                # constraints 条件の場合はsampleがその条件を満たすかどうか判定する
                func = rule["constraints"]
                if func(new_sample) == True:
                    pass
                else:
                    pass_rules = False
                    if repeat == True:
                        # sampleが条件を満たすまで、サンプリングを繰り返す
                        break
                    else:
                        # sampleが条件を満たさない場合、サンプリングを失敗し,Noneを返す
                        return None             

    return new_sample

def apply_rules_to_samples(samples, rules=None,*, n_jobs=-1):
    ''' 与えられたsamplesに対してランダムサンプリングを行う関数
    Parameters
    ----------
    samples : list or iterator or generator of [np.ndarray shape[N,1]]
        The list of samples        
    rule: list of dictionary
        the rule of sampling
         do : Function to match the value of each element to the condition
         constraints : Function describing the conditions that random sampling should satisfy
    Notes
    ----------
    samplesに対して、rulesを適用する関数
    予め、GridSearchや実験計画法などで生成したsamplesなどを引数として用いたい場合はこの関数を用いる
    またメモリ空間の節約のためにsamplesにgeneratorも受け取ることができる
    '''
    samples = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(apply_rules_to_sample)(sample, rules=rules) for sample in samples)
    return samples

def apply_rules_to_n_samples(sample, rules=None, n_samples=1000, *, n_jobs=-1):
    ''' n回ランダムサンプリングを行う関数
    Parameters
    ----------
    samples : np.ndarray shape[N,1]
        The list of samples        
    rule: list of dictionary
        the rule of sampling
         do : Function to match the value of each element to the condition
         constraints : Function describing the conditions that random sampling should satisfy
    Notes
    ----------
    同条件で繰返しランダムサンプリングを行いたい場合、n_samplesを明示的に示したい場合に使う関数    
    '''

    samples = (sample for _ in range(n_samples))
    samples = apply_rules_to_samples(samples, rules=rules, n_jobs=n_jobs)
    return samples

def generate_value_by_uniform(sample):
    ''' 一様分布に基づく乱数を取得する関数
    Parameters
    ----------
    sample : ndarray shape[N,1]
        The values of sample

    Return
    ----------
    sample : ndarray shape[N,1]
        Uniform random numbers.

    '''
    
    n_sample = len(sample)
    sample = np.random.random_sample(n_sample)
    return sample

def generate_value_by_dirichlet(sample, alpha=None, magnification=10):
    ''' ディリクレ分布に基づきランダムな値を取得する関数
        Generate a random value based on the Dirichlet distribution
    Parameters
    ----------
    sample : ndarray shape[N,1]
        The values of sample

    alpha : list of int | float [N]                         
        Parameter of dirichlet distribution

    magnification : float
        The magnification of alpha

    Returns
    ----------
    sample : ndarray shape[N,1]

    Examples
    --------
    >>> sample = np.random.random_sample(5)
    >>> alpha = [1,1,1,1,1]
    >>> generate_value_by_dirichlet(sample, alpha, magnification=10)  
        [0.1548 0.4150 0.0888 0.2369  0.1044]      
    >>> generate_value_by_dirichlet(sample, alpha, magnification=1000000)  
        # magnificationを大きくすると分布が狭くなる
        [0.20003 0.19917 0.20006  0.20054  0.20018]  
    >>> alpha = [1,1,1,1,6]
    >>> generate_value_by_dirichlet(sample, alpha, magnification=10)  
        # 出力される分布はalphaを反映する
        [0.087 0.134 0.057 0.139 0.581]
    '''
    # alphaが設定されていない場合、全て1とする
    if alpha is None:
        alpha = [1] * len(sample)
    # alphaの次元数が説明変数の数と異なる場合、全て1とする
    if len(sample) != len(alpha):
        alpha = [1] * len(sample)

    alpha = np.array(alpha)
    alpha = alpha / sum(alpha)  * magnification
    sample = stats.dirichlet.rvs(alpha, 1)
    return sample

def generate_value_by_choice(sample, candidates :list):
    ''' 与えられた候補(candidates)からランダムに値を取得する関数
    Parameters
    ----------
    sample : ndarray shape[N,1]
        The values of sample
    candidates : list of candidates [N]                         
        Parameter of dirichlet distribution

    Returns
    ----------
    sample : ndarray shape[N,1]

    Examples
    --------
    >>> sample = np.random.random_sample(5)
    >>> candidates = [
                      [1,2,3],
                      [4,5],
                      [6],
                      [7,8],
                      []
                      ]
    >>> sample = generate_value_by_choice(sample, candidates)
    >>> sample = [1, 4, 6, 8, 0.312]

    '''
    if len(sample) != len(candidates):
        raise ValueError

    # 候補がある場合 (if i != [])は、選択肢からランダムに選択する
    # 候補がない場合 (else)は、[0,1)の一様乱数を取得する
    sample = [np.random.choice(candidate,1)[0] if candidate != [] else np.random.uniform(0,1) for candidate in candidates]

    return sample

def set_sum_to_1(sample, idxs):
    ''' 指定したインデックスの合計を1とする関数
    Parameters
    ----------
    sample : ndarray shape[N,1]
        The values of sample
    idxs : list[int]
        The indexes whose sum you want to adjust to 1

    Return
    ----------
    sample : ndarray shape[N,1]

    Examples
    --------
    >>> sample = np.random.random_sample(5)
        [0.25, 0.12, 0.43, 0.11, 0.94]
    >>> idxs = [0,1,3] 
    >>> sample = set_sum_to_1(sample, idxs)
        [0.207, 0.640, 0.273, 0.153, 0.905]
    >>> sum(sample[idxs])
        1.0
    '''

    sum_idx = sum([i for idx, i in enumerate(sample) if idx in idxs])
    sample = [i/sum_idx if idx in idxs else i for idx, i in enumerate(sample)]
    return sample

def set_lower_bound(sample, idx, minimum):
    ''' 指定したインデックスの値に下限を適用する関数
        Sets the lower bound for the given index
    Parameters
    ----------
    sample : ndarray shape[N,1]
        The values of sample
    idx : int                         
        The index to apply the lower bound
    Returns
    ----------
    sample : ndarray shape[N,1]
    '''
    sample[idx] = max(sample[idx], minimum)
    return sample

def set_upper_bound(sample, idx, maximum):
    ''' 指定したインデックスの値に上限を適用する関数
        Sets the upper bound for the given index

    Parameters
    ----------
    sample : ndarray shape[N,1]
        The values of sample
    idx : int                         
        The index to apply the upper bound
    Returns
    ----------
    sample : ndarray shape[N,1]
    '''
    sample[idx] = min(sample[idx], maximum)
    return sample

def set_bounds(sample, idx , minimum=None, maximum=None, mode="resample"):
    ''' 指定した上限、下限の範囲内に値を変更する関数
    Parameters
    ----------
    sample : ndarray shape[N,1]
        The values of sample
    idx : int                         
        Index of the sample
    minimum : float
        Minimum of value
    manxmum : float
        Maximum of value
    mode : str
        Mode of set_bounds
        resample : Generate a new random value between minimum, maximum
                     元の値に関わらず、上限と下限の範囲内で新しく一様乱数を設定するモード
                     When minimum is 0.3 maximum is 0.5.
                     Regardless of the original value, new value is np.random.uniform(minimum, maximum) 

        applylimit : Apply the limit to value
                     元の値が上限、下限の範囲外であった場合は、上限値もしくは下限値を新しい値とする
                     元の値が上限、下限の範囲内であった場合は、元の値を採用する                     
                     When the value is 0.7, minimum is 0.3 maximum is 0.5
                     Then new value is max(min(value, maximum),minimum) = 0.5

                     When the value is 0.4, minimum is 0.3 maximum is 0.5
                     Then new value is max(min(value, maximum),minimum) = 0.4

    Returns
    ----------
    sample : ndarray shape[N,1]

    Examples
    --------
    >>> sample = np.random.random_sample(5)
    >>> set_bounds(sample, idx=1, minimum=0.1, maximum=0.3, mode="resample") 

    '''
    if mode not in ["resample", "applylimit"]:
        mode = "resample"

    if mode == "resample":
        if minimum is None:
            minimum = sample[idx]
        if maximum is None:
            maximum = sample[idx]

        sample[idx] = np.random.uniform(minimum, maximum)
    
    elif mode == "applylimit":
        if minimum is not None:
            sample[idx] = max(sample[idx], minimum)
        if maximum is not None:
            sample[idx] = min(sample[idx], maximum)

    return sample 

def set_ratio_range(sample, idx, minimum_ratio=0.0, maximum_ratio=1.0):
    ''' 指定したインデックスの値が、指定の範囲内の比率となり、かつsampleの合計値が1.0となるように設定する関数

    Parameters
    ----------
    sample : ndarray shape[N,1]
        The values of sample
    idx : int                         
        Index of the sample
    minimum_ratio : float
        Minimum of value
    manxmum_ratio : float
        Maximum of value

    Returns
    ----------
    sample : ndarray shape[N,1]

    Examples
    --------
    >>> sample = np.random.random_sample(5)
        [0.15, 0.35, 0.12, 0.94, 0.24]
    >>> idx = 0
    >>> sample = set_ratio_range(sample, idx, minimum_ratio=0.3, maximum_ratio=0.6)     
        [0.3266, 0.1428, 0.048, 0.3836, 0.097]
    >>> sum(sample)
        1.0

    Notes
    --------
    組成系のランダムサンプリングなどで、合計を1.0とする制限下で
    1つのの組成の比率の上限、下限を設定できる
    
    複数の組成の比率を設定したい場合はset_ratio_ranges関数を用いる

    '''
    # minimum_ratioの下限値を0.0とする
    minimum_ratio = max(minimum_ratio, 0.0)
    # maximum_ratioの上限値を1.0とする
    maximum_ratio = min(maximum_ratio, 1.0)
    # 比率の範囲内で一様乱数を取得する
    new_ratio = np.random.uniform(minimum_ratio, maximum_ratio)    
    # 指定したインデックス以外の合計値を計算する
    other_ratios = sum([ratio for idx_, ratio in enumerate(sample) if idx_ != idx])
    # sampleの合計を1.0とするため、指定したインデックス以外を定数倍する倍率を求める
    magnification = (1.0 - new_ratio) / other_ratios
    # 指定したインデックスの値を設定する
    sample[idx] = new_ratio
    # 指定したインデックス以外の値を定数倍する
    sample = [ratio * magnification if idx_ != idx else ratio for idx_, ratio in enumerate(sample) ]
    return sample

def set_ratio_ranges(sample, group_idxs :list, settings :list, * , adjust_maximum=False): 
    ''' 
    Parameters
    ----------
    sample : ndarray shape[N,1]
        The values of sample
    group_idxs : list of int                         
        Index of groups with a sum of 1
    settings : list of dict
        A dictionary of indexes that specify the maximum and minimum ratios.

    Returns
    ----------
    sample : ndarray shape[N,1]

    Examples
    --------
    >>> sample = np.random.random_sample(5)
    >>> group_idxs = [0,1,2]
    >>> settings = [{"idx" : 0,
                    "minimum_ratio" : 0.0,
                    "maximum_ratio" : 1.0,
                    },
                    {"idx" : 1,
                    "minimum_ratio" : 0.1,
                    },
                    {"idx" : 2,
                    "maximum_ratio" : 0.8,
                    },
                    ]
    >>> set_ratio_ranges(sample, group_idxs, settings)        

    Notes
    --------
    説明変数に組成系(index:0,1,2)と実験条件(index:3,4)などが含まれる場合のランダムサンプリングにおいて
    組成系(index:0,1,2)の合計は1.0としたい場合に用いる関数
    settingsに指定することで、それぞれの組成の比率も指定することができる
    
    あるいは組成系A(index:0,1,2)と組成系(index:3,4)など、それぞれで合計を1.0にしたい場合にも用いることが出来る
    '''

    target_indexes = []
    sum_new_ratios = 0.0
    for setting in settings:
        target_index = setting.get("idx", None)
        if target_index is None:
            continue        
        if target_index not in group_idxs:
            continue
        target_indexes.append(target_index)
        minimum_ratio = setting.get("minimum_ratio", 0.0)
        maximum_ratio = setting.get("maximum_ratio", 1.0)
        if adjust_maximum == True:
            maximum_ratio = 1.0 - sum_new_ratios
            if maximum_ratio <= minimum_ratio:
                maximum_ratio = minimum_ratio
            print(f"maximum_ratio {maximum_ratio}")

        new_ratio = np.random.uniform(minimum_ratio, maximum_ratio)    
        sum_new_ratios += new_ratio
        sample[target_index] = new_ratio

    other_ratios = sum([ratio for idx_, ratio in enumerate(sample) if (idx_ not in target_indexes) and (idx_ in group_idxs)])
    if other_ratios > 0:
        if sum_new_ratios < 1.0:
            other_magnification = (1 - sum_new_ratios) / other_ratios
            sample = [ratio * other_magnification if (idx_ not in target_indexes) and (idx_ in group_idxs) else ratio for idx_, ratio in enumerate(sample) ]
        elif sum_new_ratios >= 1.0:
            # 組成比を指定された組成の合計が1を超えた場合は、未指定の組成を0とし、指定された組成の合計を1.0に変換する
            new_magnification = 1 / sum_new_ratios
            sample = [0 if (idx_ not in target_indexes) and (idx_ in group_idxs) else ratio * new_magnification for idx_, ratio in enumerate(sample) ]
    elif other_ratios == 0:
        sample = [ratio / sum_new_ratios if idx_ in target_indexes else ratio for idx_, ratio in enumerate(sample)]

    return sample

def set_exclusive(sample, idxs:list, k:int = 1):
    ''' 指定したインデックス群から、k個のインデックスのみ値を採用する関数
    Parameters
    ----------
    sample : ndarray shape[N,1]
        The values of sample
    idxs : list of int                         
        Index of a group whose elements are mutually exclusive values
    k : int
        Number of elements in a group that are allowed to have a value greater than or equal to 0

    Returns
    ----------
    sample : ndarray shape[N,1]

    Examples
    --------
    >>> sample = np.random.random_sample(5)
        [0.4, 0.47, 0.91, 0.28, 0.49]
    >>> idxs = [[0,1,2],[3,4]]
        インデックス群(0, 1, 2)からk個のインデックスの値を採用し、他のインデックスの値は0とする 
        インデックス群(3, 4)からk個のインデックスの値を採用し、他のインデックスの値は0とする 
    >>> set_exclusive(sample, idxs, k=1)  
        [0.4, 0, 0, 0.28, 0] 
    >>> set_exclusive(sample, idxs, k=2)  
        [0, 0.47, 0.91, 0.28, 0.49] 
    
    Notes
    --------
    組成系のランダムサンプリングなどで、材料Ａと材料Ｂのどちらかしか使えない場合など
    排反的に組成を設計する際に使う関数

    '''
    for exclusive_idxs in idxs:
        selected_idx = random.sample(exclusive_idxs, k)
        unselected_idx = [i for i in idxs if i not in selected_idx]
        sample = [i  if idx not in unselected_idx else 0 for idx, i in enumerate(sample)]

    return sample

if __name__ == "__main__":

    # 例1
    # 5次元
    # ランダムサンプリング

    n_X = 5
    sample = np.zeros(n_X)
    rules = [
            #各組成をランダムに設定
            {"do": lambda i: generate_value_by_uniform(i)},
            {"do": lambda i: set_bounds(i, 0, minimum=0.3, maximum=0.5, mode="resample")},
            {"do": lambda i: set_bounds(i, 1, minimum=0.3, mode="resample")},
            {"do": lambda i: set_bounds(i, 2, minimum=0.3, maximum=0.5, mode="applylimit")},
            {"do": lambda i: set_bounds(i, 3, minimum=0.3, mode="applylimit")},
            ]
    random_samples = apply_rules_to_n_samples(sample, rules, n_samples=20)
    pprint.pprint(random_samples)
    print(sample)


    # 例2
    # 5次元
    # ランダムサンプリング
    # 排他的に変数を選択
    # 変数の合計を1に規格化

    n_X = 5
    sample = np.zeros(n_X)
    exclusive_idxs = [[1,2],[3],[0]]
    rules = [
            # 各組成をランダムに設定
            {"do": lambda i: generate_value_by_uniform(i)},
            # 各組成を排他的に設定
            {"do": lambda i: set_exclusive(i, exclusive_idxs,2)},
            # 組成の和を1とする
            {"do": lambda i: i/sum(i)}, 
            ]
    # random_sample = apply_rules_to_sample(sample, rules)
    random_samples = apply_rules_to_n_samples(sample, n_samples=10, rules=rules)
    print(sample)

    # 例3
    # 5次元
    # ランダムサンプリング
    # 組成1の最小割合、最大割合を設定(0.7 <= 0.9)    
    # 変数の合計を1に規格化

    n_X = 5
    sample = np.zeros(n_X)
    rules = [
            # 各組成をランダムに設定する
            {"do": lambda i: generate_value_by_uniform(i)},
            # 組成和を1とする
            {"do": lambda i: i/sum(i)}, 
            # 組成1の最小割合、最大割合を満たすように設定
            {"do": lambda i: set_ratio_range(i, 1, minimum_ratio=0.7, maximum_ratio=0.9)}, 
            # 組成3が0.1以上、組成4が0.5以下の制約条件を課す
            {"constraints": lambda i: i[3]>=0.1 and i[4]<=0.5},
            ]
    
    start_time = time.time()
    random_samples = apply_rules_to_n_samples(sample, n_samples=5, rules=rules)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)
    print(random_samples[:3])


    # 例4
    # 7次元：組成系A(0,1,2)、組成系B(3,4)、温度(5),、圧力(6)
    # ランダムサンプリング
    # グループA(0,1,2)の合計を1に規格化,グループB(3,4) の合計を1に規格化
    # 組成1の最小割合、最大割合の制約条件を設定(0.7 <= 0.9) 
    # 組成3,組成4の制約条件を加える   
    # 組成0,組成2の制約条件を加える   
    n_X = 7
    sample = np.zeros(n_X)
    group_idxs_1 = [0,1,2]
    group_idxs_2 = [3,4]
    
    rules = [
            {"do": lambda i: np.zeros(7)},
            # 各組成をランダムに設定する            
            {"do": lambda i: generate_value_by_uniform(i)},
            # グループ1の組成和を1とする
            {"do": lambda i: set_sum_to_1(i, group_idxs_1)}, 
            # グループ1の組成和を2とする
            {"do": lambda i: set_sum_to_1(i, group_idxs_2)}, 
            # 温度の範囲を25-30とする
            {"do": lambda i: set_bounds(i, 5 , minimum=25, maximum=30)}, 
            # 圧力の範囲を1-5とする
            {"do": lambda i: set_bounds(i, 6 , minimum=1.0, maximum=5)}, 
            # 組成1の範囲を制約条件を加える。確率が低いので推奨しない
            {"constraints": lambda i: 0.7 <= i[1] <=0.9},
            # 組成3,組成4の制約条件を加える
            {"constraints": lambda i: i[3]>=0.1 and i[4]<=0.5},
            # 組成0と組成2の和の制約条件を加える
            {"constraints": lambda i: (i[3] + i[2])<0.9},
            ]
    
    start_time = time.time()
    random_samples = apply_rules_to_n_samples(sample, n_samples=5, rules=rules)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)
    print(random_samples[:3])

    # 例5
    # 5次元
    # ディリクレ分布に基づくランダムサンプリング

    alpha = [5,5,10,20,60]
    rules = [
            # 各組成をディリクレ分布に基づきランダムに設定
            {"do": lambda i: generate_value_by_dirichlet(i, alpha, magnification=5)},
            ]
    random_sample = apply_rules_to_sample(sample, rules)
    print('dirichlet')
    print(random_sample)

    start_time = time.time()
    random_samples = apply_rules_to_n_samples(sample, n_samples=1, rules=rules)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)
    print(random_samples[:3])

    # 例6

    n_X = 5
    sample = np.zeros(n_X)
    group_idxs_1 = [0,1,2]
    group_idxs_2 = [3,4]

    idx_range_1 = [{"idx" : 0, "minimum_ratio" : 0.4, "maximum_ratio" : 0.8},
                   {"idx" : 1, "minimum_ratio" : 0.4},
                   {"idx" : 2, "minimum_ratio" : 0.2},
                    ]   

    idx_range_2 = [{"idx" : 3, "minimum_ratio" : 0.3, "maximum_ratio" : 1.0}
                    ]   

    rules = [
            # 各組成をランダムに設定する
            {"do": lambda i: generate_value_by_uniform(i)},
            # グループ1の組成割合を反映する
            {"do": lambda i: set_ratio_ranges(i, group_idxs_1, idx_range_1, adjust_maximum=True)}, 
            # グループ2の組成割合を反映する
            {"do": lambda i: set_ratio_ranges(i, group_idxs_2, idx_range_2)}, 
            ]
    
    start_time = time.time()
    random_samples = apply_rules_to_n_samples(sample, n_samples=10, rules=rules)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)
    pprint.pprint(random_samples[:5])

    # choices
    n_X = 5
    sample = np.zeros(n_X)
    candidates = [[1,2,3,4,5,6,7,8,9,10],
                  [4,5],
                  [6],
                  [7,8],
                  []]

    rules = [
            # 各組成をランダムに設定する
            {"do": lambda i: generate_value_by_choice(i, candidates)},
            ]
    
    random_samples = apply_rules_to_n_samples(sample, n_samples=10, rules=rules)
    print('choices')
    pprint.pprint(random_samples)

    # ディリクレ分布に基づく
    n_X = 5
    sample = np.zeros(n_X)
    alpha = [0.5  for i in range(n_X)]
    # alpha = [0.5, 0.25, 0.125, 0.0625, 0.03125]

    rules = [
            # 各組成をランダムに設定する
            {"do": lambda i: generate_value_by_choice(i, candidates)},
            ]
    

    random_samples = apply_rules_to_n_samples(sample, n_samples=10, rules=rules)
    print('choices')
    pprint.pprint(random_samples)


    # 例　全組合せに対して、制約条件を適用
    import itertools
    levels = [[0,3,6],
                [0,5,6],
                [3,4,5],
                [1],
                [1,2,3,4,5,6]]

    rules = [
            # 組成2, 組成4の合計が6以上
            {"constraints": lambda i: (i[2] + i[4])>=6},
            ]

    samples = itertools.product(*levels)
    random_samples = apply_rules_to_samples(samples, rules=rules)
    print("全組合せに対して、制約条件を適用")
    print(random_samples[:20])
    
    import more_itertools

    # 例　組合せ数が非常に多い場合に全組合せからランダムに選択した上で制約条件を適用
    # more_itertoolsを用いると、メモリ節約できる（リストではなくジェネレータを渡せるため）

    levels = [[i for i in range(40)],
              [i for i in range(40)],
              [i for i in range(40)],
              [i for i in range(40)],
              [i for i in range(40)]
              ]
    # 40^5 = 102400000
    rules = [
            # 組成2, 組成4の合計が6以上
            {"constraints": lambda i: (i[2] + i[4])<=40},
            ]

    samples = itertools.product(*levels)
    samples = more_itertools.sample(samples, 1000000)
    random_samples = apply_rules_to_samples(samples, rules=rules)
    print("組合せ数が非常に多い場合に全組合せからランダムに選択した上で制約条件を適用")
    print(random_samples[:20])