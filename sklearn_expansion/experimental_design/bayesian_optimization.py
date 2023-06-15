
import random
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    WhiteKernel,
    RBF,
    ConstantKernel,
    Matern,
    DotProduct,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


def acquisition_EI(
    y,
    y_design_predict,
    y_design_std_deviation,
    jitter,
    objective="maximize",
):
    acquisition_function = (
        y_design_predict - max(y) - jitter * y.std()
    ) * norm.cdf(
        (y_design_predict - max(y) - jitter * y.std())
        / y_design_std_deviation
    ) + y_design_std_deviation * norm.pdf(
        (y_design_predict - max(y) - jitter * y.std())
        / y_design_std_deviation
    )
    return acquisition_function.reshape(-1,1)

def acquisition_PI(y, y_design_predict, y_design_std_deviation, jitter):
    acquisition_function = norm.cdf(
        (y_design_predict - max(y) - jitter * y.std()) / y_design_std_deviation
    )
    return acquisition_function

def acquisition_PTR(y_design_predict, y_design_std_deviation, lower, upper):
    acquisition_function = norm.cdf(
        upper, loc=y_design_predict, scale=y_design_std_deviation
    ) - norm.cdf(lower, loc=y_design_predict, scale=y_design_std_deviation)
    return acquisition_function

class BayesianOptimization:
    """
    """
    def __init__(self):
        pass
    
    @staticmethod
    def kernels():
        _kernels = [
        ConstantKernel() * DotProduct() + WhiteKernel(noise_level_bounds=[1e-7, 1e-1]),
        ConstantKernel() * RBF() + WhiteKernel(noise_level_bounds=[1e-7, 1e-1]),
        ConstantKernel() * RBF() + WhiteKernel(noise_level_bounds=[1e-7, 1e-1]) + ConstantKernel() * DotProduct(),
        ConstantKernel() * RBF() + WhiteKernel(noise_level_bounds=[1e-7, 1e-1]),
        ConstantKernel() * RBF() + WhiteKernel(noise_level_bounds=[1e-7, 1e-1]) + ConstantKernel() * DotProduct(),
        ConstantKernel() * Matern(nu=1.5) + WhiteKernel(noise_level_bounds=[1e-7, 1e-1]),
        ConstantKernel() * Matern(nu=1.5) + WhiteKernel(noise_level_bounds=[1e-7, 1e-1]) + ConstantKernel() * DotProduct(),
        ConstantKernel() * Matern(nu=0.5) + WhiteKernel(noise_level_bounds=[1e-7, 1e-1]),
        ConstantKernel() * Matern(nu=0.5) + WhiteKernel(noise_level_bounds=[1e-7, 1e-1]) + ConstantKernel() * DotProduct(),
        ConstantKernel() * Matern(nu=2.5) + WhiteKernel(noise_level_bounds=[1e-7, 1e-1]),
        ConstantKernel() * Matern(nu=2.5) + WhiteKernel(noise_level_bounds=[1e-7, 1e-1]) + ConstantKernel() * DotProduct(),
        ]
        return _kernels

    def kriging_believer_algorithm(
        self,
        estimator,
        X,
        y,
        n_y,
        X_design,
        acquisition_function="PTR",
        target_ranges=None,
        jitter=0.05,
        n_suggestion=3,
        *,
        objective="maximize",
        random_state=None,
        verbose=False,
        jitter_fluctuation=True,
    ):
        """Kriging Believer Algorithm

        Parameters
        ----------
        estimator : sklearn.estimator
            Estimator needs the attribute(return_std)
        X : np.ndarray
            Feature vectors or other representations of training data.
        y : np.ndarray
            Target values.
        X_design : np.ndarray #TODO (or iterator)
            Design space of X
        acquisition_function : str
            The function of bayesian optimization.
            "PTR", "EI", "PI"
        target_ranges : dict
            Target ranges for PTR.
        jitter : float
            Positive value to make the acquisition more explorative.

        n_suggestion : int, optional
            Number of suggestion of Kriging Believer Algorithm, by default 1.
        objective : list[string] or string
            Direction of optimization.
        random_state : int

        verbose : Boolean

        Returns
        -------
        next_Xs : np.array
            Next candidate for experiment.

        Examples
        -------
        >>> X, y = load_diabetes(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=10)
        >>> n_y = 1
        >>> kernel = BayesianOptimization.kernels()[4]
        >>> estimator = GaussianProcessRegressor(alpha=0,
                                                        kernel=kernel,
                                                        random_state=None,
                                                        optimizer='fmin_l_bfgs_b')

        >>> target_ranges = [{"lower":250, "upper":np.inf}]
        >>> bo = BayesianOptimization()
        >>> next_Xs = bo.kriging_believer_algorithm(estimator=estimator,
                                                    X=X_train,
                                                    y=y_train,
                                                    n_y=n_y,
                                                    X_design=X_test,
                                                    target_ranges=target_ranges,
                                                    acquisition_function="PTR",
                                                    n_suggestion=n_suggestion)
        """
        np.random.seed(random_state)


        X = np.array(X)
        y = np.array(y).reshape(-1, n_y)
        X_design = np.array(X_design)

        self.init_X = X
        self.init_y = y

        if isinstance(objective, str):
            objective = [objective] * n_y

        # target_rangesが設定されていない場合は下限をyの最大値以上（最大化問題）として設定する
        if target_ranges is None:
            target_ranges = []
            for i in range(n_y):
                target_ranges.append({"lower": max(y[:, i]), "upper": np.inf})

        # 下限、上限が指定されていない場合は、それぞれ-np.inf, np.infに設定する        
        for each_target_range in target_ranges:
            if each_target_range.get("lower", None) is None:
                each_target_range["lower"] = -np.inf
            if each_target_range.get("upper", None) is None:
                each_target_range["upper"] = np.inf

        self.target_ranges = target_ranges

        # jitterの値をランダムに設定する
        if jitter_fluctuation==True:
            if random.random() <= 0.3:
                jitter = 10

        # 結果格納用リスト
        next_Xs = []
        next_Xs_scaled = []
        next_ys = []
        next_ys_std_deviation = []
        next_acquisition_values = []

        X_scaler = StandardScaler()
        y_scaler = StandardScaler()

        # ガウス過程回帰
        model = estimator
        
        if n_y >= 2:
            model = MultiOutputRegressor(model)

        for i in range(n_suggestion):
            # 学習データをスケーリング
            X_scaled = X_scaler.fit_transform(X)
            y_scaled = y_scaler.fit_transform(y.reshape(-1, n_y))

            # 実験空間をスケーリング
            X_design_scaled = X_scaler.transform(X_design)

            for i in range(n_y):
                model.fit(X_scaled, y_scaled)

                # 実験空間の予測値、分散
                if n_y >= 2:
                    for i in range(n_y):
                        (
                            each_y_design_predict_scaled,
                            each_y_design_std_deviation_scaled,
                        ) = model.estimators_[i].predict(
                            X_design_scaled, return_std=True
                        )
                        each_y_design_predict_scaled = (
                            each_y_design_predict_scaled.reshape(-1, 1)
                        )
                        each_y_design_std_deviation_scaled = (
                            each_y_design_std_deviation_scaled.reshape(-1, 1)
                        )

                        if i == 0:
                            y_design_predict_scaled = each_y_design_predict_scaled
                            y_design_std_deviation_scaled = (
                                each_y_design_std_deviation_scaled
                            )
                        else:
                            y_design_predict_scaled = np.concatenate(
                                [y_design_predict_scaled, each_y_design_predict_scaled],
                                axis=1,
                            )
                            y_design_std_deviation_scaled = np.concatenate(
                                [
                                    y_design_std_deviation_scaled,
                                    each_y_design_std_deviation_scaled,
                                ],
                                axis=1,
                            )

                else:
                    (
                        y_design_predict_scaled,
                        y_design_std_deviation_scaled,
                    ) = model.predict(X_design_scaled, return_std=True)

                    y_design_predict_scaled = y_design_predict_scaled.reshape(-1, 1)
                    y_design_std_deviation_scaled = (
                        y_design_std_deviation_scaled.reshape(-1, 1)
                    )

                y_design_predict_raw = y_scaler.inverse_transform(
                    y_design_predict_scaled
                )
                y_design_std_deviation_raw = (
                    y_design_std_deviation_scaled * y_scaler.scale_
                )

                if acquisition_function == "EI":
                    acquisition_values = np.zeros([X_design.shape[0], 1])
                elif acquisition_function == "PI":
                    acquisition_values = np.ones([X_design.shape[0], 1])
                elif acquisition_function == "PTR":
                    acquisition_values = np.ones([X_design.shape[0], 1])
                else:
                    warnings.warn(
                        f"acquisition_function {acquisition_function} is not defined."
                    )
                    acquisition_function = "PTR"
                    acquisition_values = np.ones([X_design.shape[0], 1])

                each_acquisition_values = []
                for i in range(n_y):
                    # 獲得関数の計算
                    if acquisition_function == "EI":
                        if objective[i] == "maximize":
                            acquisition_function_design = acquisition_EI(
                                y=y[:, i],
                                y_design_predict=y_design_predict_raw[:, i].ravel(),
                                y_design_std_deviation=y_design_std_deviation_raw[
                                    :, i
                                ].ravel(),
                                jitter=jitter,
                                )

                            acquisition_function_design /= y[:, i].std()

                        elif objective[i] == "minimize":
                            acquisition_function_design = acquisition_EI(
                                y=-y[:, i],
                                y_design_predict=-y_design_predict_raw[:, i].ravel(),
                                y_design_std_deviation=y_design_std_deviation_raw[
                                    :, i
                                ].ravel(),
                                jitter=jitter)
                            acquisition_function_design /= y[:, i].std()
                        else:
                            raise ValueError(f"{objective[i]}にはmaximize, minimizeのいずれかを選んでください")


                        acquisition_values = (
                            acquisition_values + acquisition_function_design
                        )

                    elif acquisition_function == "PI":
                        acquisition_function_design = acquisition_PI()
                        acquisition_values = (
                            acquisition_values * acquisition_function_design
                        )

                    elif acquisition_function == "PTR":
                        lower = target_ranges[i].get("lower", -np.inf)
                        upper = target_ranges[i].get("upper", np.inf)

                        acquisition_function_design = acquisition_PTR(
                            y_design_predict=y_design_predict_raw[:, i].ravel(),
                            y_design_std_deviation=y_design_std_deviation_raw[
                                :, i
                            ].ravel(),
                            lower=lower,
                            upper=upper,
                        )
                        acquisition_function_design = (
                            acquisition_function_design.reshape(-1, 1)
                        )
                        acquisition_values = (
                            acquisition_values
                            * acquisition_function_design.reshape(-1, 1)
                        )

                    each_acquisition_values.append(acquisition_function_design)

            if False:
                noise = np.random.uniform(low=0.00, high=0.01, size=acquisition_values.shape)
                print(acquisition_values.shape)
                print(acquisition_values[0])
                acquisition_values =  acquisition_values +  noise

            next_idx = np.argmax(acquisition_values)
            next_acquisition_value = acquisition_values[next_idx]
            next_acquisition_values.append(next_acquisition_value)

            X_next_raw = X_design[next_idx, :].reshape(1, -1)

            # 予測用のscaled X
            X_next_scaled = X_scaler.transform(X_next_raw)

            # 予測用Xから次の実験行を削除して、次の予測に対応させる
            X_design = np.delete(X_design, next_idx, axis=0)

            # 予測Xを格納
            next_Xs.append(X_next_raw)
            next_Xs_scaled.append(X_next_scaled)

            # 次の実験点での期待値、標準偏差を取得(scaled)
            y_next_scaled = y_design_predict_scaled[next_idx]
            y_next_raw = y_design_predict_raw[next_idx]
            y_next_std_deviation = y_design_std_deviation_raw[next_idx]

            # 予測結果をもとにモデルを再構築するため、yをスケーリング前に戻してから追加
            next_ys.append(y_next_raw)
            next_ys_std_deviation.append(y_next_std_deviation)

            # 次のモデル構築のために、予測した行を追加 Kriging-Believer
            X = np.append(X, X_next_raw, axis=0)
            y = np.append(y, y_next_raw.reshape(-1, n_y), axis=0)

        next_Xs = np.array(next_Xs).reshape(n_suggestion, -1)
        self.next_Xs = next_Xs
        self.next_ys = np.array(next_ys).reshape(n_suggestion, -1)
        self.next_ys_std_deviation = np.array(next_ys_std_deviation).reshape(n_suggestion, -1)
        self.next_acquisition_values = np.array(next_acquisition_values).reshape(n_suggestion, -1)
        self.X_design = X_design

        return next_Xs

    def run_bayesian_optimization(
        self,
        estimator,
        X,
        y,
        n_y,
        X_design,
        target_ranges=None,
        jitter=0.05,
        acquisition_function="PTR",
        n_suggestion=3,
        n_optimize=1,
        func=None,
        *,
        jitter_fluctuation=True,
        objective="maximize",
        random_state=None,
        verbose=False,
        multipleproposal="kriging-believer",
    ):
        """BayesianOptimization

        Parameters
        ----------
        X : np.ndarray
            Feature vectors or other representations of training data.
        y : np.ndarray
            Target values.
        X_design : np.ndarray #TODO (or iterator)
            Design space of X
        n_suggestion : int, optional
            Number of cycle of Kriging Believer Algorithm, by default 1.
        n_optimize : int, optional
            Number of cycle of bayesian optimization, by default 1
        func : function or None, optional
            A function that computes Y from X or None.
            If function is None, The value of n_optimize is forced to 1.
        objective : list[string] or string
            Direction of optimization.

        Returns
        -------
        _type_
            _description_
        """


        self.original_X = np.array(X)
        self.original_y = np.array(y).reshape(-1, n_y)

        self.X = np.array(X)
        self.y = np.array(y).reshape(-1, n_y)
        self.n_y = n_y
        self.X_design = X_design

        if isinstance(objective, str):
            objective = [objective] * n_y


        next_Xs = []
        next_Xs_scaled = []
        next_ys = []
        next_ys_std_deviation = []

        for i in range(n_optimize):
            if verbose == True:
                print(f"optimize {i}")
                print(f"n_suggestion {n_suggestion}")

            (
                kriging_next_Xs,
                kriging_next_ys,
                kriging_next_ys_std_deviation,
                next_X_design,
            ) = self.kriging_believer_algorithm(
                estimator,
                self.X,
                self.y,
                n_y,
                X_design=self.X_design,
                target_ranges=target_ranges,
                jitter=jitter,
                n_suggestion=n_suggestion,
                acquisition_function=acquisition_function,
                random_state=random_state,
                objective=objective,
                jitter_fluctuation=jitter_fluctuation,
            )

            next_Xs.append(kriging_next_Xs)
            next_ys.append(kriging_next_ys)
            next_ys_std_deviation.append(kriging_next_ys_std_deviation)

            if func is None:
                break
            elif len(next_X_design) == 0:
                break
            else:
                # print(np.array(kriging_next_Xs))
                computed_Y = func(np.array(kriging_next_Xs).reshape(n_suggestion, -1))
                computed_Y = np.array(computed_Y).reshape(-1, n_y)
                if verbose == True:
                    print(f"computed_Y {computed_Y}")

                self.X = np.concatenate(
                    [self.X, np.array(kriging_next_Xs).reshape(n_suggestion, -1)], axis=0
                )
                self.y = np.concatenate([self.y, np.array(computed_Y)], axis=0)
                self.X_design = next_X_design
            
            self.next_Xs = next_Xs
            self.next_ys = next_ys
            self.next_ys_std_deviation = next_ys_std_deviation

        return next_Xs