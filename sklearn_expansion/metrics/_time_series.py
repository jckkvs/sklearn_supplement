import numpy as np

def r2lm_score(y_true, y_pred,*, moving_window=1, multioutput="uniform_average"):        
    """
    r2 based on the latest measured y-values
    https://datachemeng.com/r2lm/ に基づいて計算
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
        y_trueとy_predはそれぞれ同じ時系列順でソートされていることを前提とする
    moving_window : int
        Ground truth (correct) target values.
    moving_window : {'raw_values', 'uniform_average', 'variance_weighted'}, \
                        array-like of shape (n_outputs,) or None, default='uniform_average'
        'raw_values' :
            Returns a full set of scores in case of multioutput input.
        'uniform_average' :
            Scores of all outputs are averaged with uniform weight.
        'variance_weighted' :
            Scores of all outputs are averaged, weighted by the variances
            of each individual output.

    Returns
    ----------
    r2lm_score : float or ndarray of floats
        The :math:`R^2` score or ndarray of scores if 'multioutput' is
        'raw_values'.


    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 目的変数の数を取得
    if len(y_pred.shape) == 1:
        outputs_num = 1
    elif len(y_pred.shape) == 2:
        outputs_num = y_pred.shape[1]

    denominator = np.zeros(outputs_num)

    # 分母の計算

    for sample_index, _ in enumerate(y_true):
        if sample_index <= (moving_window-1):
            continue

        each_y_true = y_true[sample_index]
        moving_average_y_previous_true = y_true[sample_index-moving_window : sample_index]    

        each_y_true_np     = np.array(each_y_true)
        moving_average_y_previous_true_np = np.array(moving_average_y_previous_true).mean(axis=0)
        denominator   += (each_y_true_np - moving_average_y_previous_true_np)**2

    #　分子の計算
    numerator = ((y_true - y_pred)** 2).sum(axis=0)

    r2lm_each_score = 1- (numerator/denominator)

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            # return scores individually
            return output_scores

        elif multioutput == "uniform_average":
            # passing None as weights results is uniform mean
            avg_weights = None
        elif multioutput == "variance_weighted":
            avg_weights = denominator
            # avoid fail on constant y or one-element arrays
            if not np.any(nonzero_denominator):
                if not np.any(nonzero_numerator):
                    return 1.0
                else:
                    return 0.0
    else:
        avg_weights = multioutput

    return np.average(r2lm_each_score, weights=avg_weights)

