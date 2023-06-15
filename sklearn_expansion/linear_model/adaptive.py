import sys
import functools
import itertools
import logging
import multiprocessing as mp
import joblib

import cvxpy
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
)
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import _num_features, _num_samples


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class BaseModel(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        alpha=1.0,
        *,
        penalization=None,
        quantile=False,
        adaptive=False,
        group=False,
        sparse=False,
        fit_intercept=True,
        tol=0.0001,
        lambda1=1.0,
        tau=0.5,
        lasso_weights=None,
        group_lasso_weights=None,
        n_jobs=None,
        solver="default",
        max_iter=500,
    ):
        """
        Parameters:
            model: model to be fit (accepts 'lm' or 'qr')
            penalization: penalization to use (accepts None, 'lasso', 'gl', 'sgl', 'asgl', 'asgl_lasso', 'asgl_gl',
                            alasso, agl)
            fit_intercept: boolean, whether to fit the model including fit_intercept or not
            tol:  tolerance for a coefficient in the model to be considered as 0
            lambda1: parameter value that controls the level of shrinkage applied on penalizations
            alpha: parameter value, tradeoff between lasso and group lasso in sgl penalization
            tau: quantile level in quantile regression models
            lasso_weights: lasso weights in adaptive penalizations
            group_lasso_weights: group lasso weights in adaptive penalizations
            parallel: boolean, whether to execute the code in parallel or sequentially
            n_jobs: if parallel is set to true, the number of cores to use in the execution. Default is (max - 1)
            solver: solver to be used by CVXPY. default uses optimal alternative depending on the problem
            max_iter: CVXPY parameter. Default is 500

        Returns:
            This is a class definition so there is no return. Main method of this class is fit,  that has no return
            but outputs automatically to _coef.
            ASGL._coef stores a list of regression model coefficients.
        """
        self.alpha = alpha
        self.penalization = penalization
        self.quantile = quantile
        self.group = group
        self.sparse = sparse
        self.adaptive = adaptive

        self.fit_intercept = fit_intercept
        self.tol = tol
        self.lambda1 = lambda1
        self.alpha = alpha
        self.tau = tau
        self.lasso_weights = lasso_weights
        self.group_lasso_weights = group_lasso_weights
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.coef_ = None

        # CVXPY solver parameters
        self.solver_stats = None
        self.solver = solver

    # CVXPY SOLVER RELATED OPTIONS ###################################################################################

    def _cvxpy_solver_options(self, solver):
        if solver == "ECOS":
            solver_dict = dict(solver=solver, max_iters=self.max_iter)
        elif solver == "OSQP":
            solver_dict = dict(solver=solver, max_iters=self.max_iter)
        else:
            solver_dict = dict(solver=solver)
        return solver_dict

    # SOLVERS #########################################################################################################

    def _quantile_function(self, X):
        """
        Quantile function required for quantile regression models.
        """
        return 0.5 * cvxpy.abs(X) + (self.tau - 0.5) * X

    def _num_beta_var_from_group_index(self, group_index):
        """
        Internal function used in group based penalizations (gl, sgl, asgl, asgl_lasso, asgl_gl)
        """
        group_sizes = []
        beta_var = []
        unique_group_index = np.unique(group_index)
        # Define the objective function
        for idx in unique_group_index:
            group_sizes.append(len(np.where(group_index == idx)[0]))
            beta_var.append(cvxpy.Variable(len(np.where(group_index == idx)[0])))
        return group_sizes, beta_var

    def unpenalized_solver(self, X, y):
        n_samples, n_features = X.shape
        # If we want an fit_intercept, it adds a column of ones to the matrix X
        if self.fit_intercept:
            n_features = n_features + 1
            X = np.c_[np.ones(n_samples), X]
        # Define the objective function
        beta_var = cvxpy.Variable(n_features)
        if self.quantile == False:
            objective_function = (1.0 / n_samples) * cvxpy.sum_squares(y - X @ beta_var)
        else:
            objective_function = (1.0 / n_samples) * cvxpy.sum(
                self._quantile_function(X=(y - X @ beta_var))
            )
        objective = cvxpy.Minimize(objective_function)
        problem = cvxpy.Problem(objective)
        # Solve the problem. If solver is left as default, try optimal solver sugested by cvxpy.
        # If other name is provided, try the name provided
        # If these options fail, try default ECOS, OSQP, SCS options
        try:
            if self.solver == "default":
                problem.solve(warm_start=True)
            else:
                solver_dict = self._cvxpy_solver_options(solver=self.solver)
                problem.solve(**solver_dict)
        except (ValueError, cvxpy.error.SolverError):
            logging.warning(
                "Default solver failed. Using alternative options. Check solver and solver_stats for more "
                "details"
            )
            solver = ["ECOS", "OSQP", "SCS"]
            for elt in solver:
                solver_dict = self._cvxpy_solver_options(solver=elt)
                try:
                    problem.solve(**solver_dict)
                    if "optimal" in problem.status:
                        break
                except (ValueError, cvxpy.error.SolverError):
                    continue
        self.solver_stats = problem.solver_stats
        if problem.status in ["infeasible", "unbounded"]:
            logging.warning("Optimization problem status failure")
        beta_sol = beta_var.value
        beta_sol[np.abs(beta_sol) < self.tol] = 0
        return [beta_sol]

    def agl(self, X, y, group_index, param):
        """
        Group lasso penalized solver
        """
        n_samples = _num_samples(X)
        # Check th group_index, find the unique groups, count how many vars are in each group (this is the group size)
        group_index = np.asarray(group_index).astype(int)
        unique_group_index = np.unique(group_index)
        group_sizes, beta_var = self._num_beta_var_from_group_index(group_index)
        num_groups = len(group_sizes)
        model_prediction = 0
        group_lasso_penalization = 0
        # If the model has an intercept, we calculate the value of the model for the intercept group_index
        # We start the penalization in inf_lim so if the model has an intercept, penalization starts after the intercept
        inf_lim = 0
        if self.fit_intercept:
            # Adds an element (referring to the intercept) to group_index, group_sizes, num groups
            group_index = np.append(0, group_index)
            unique_group_index = np.unique(group_index)
            X = np.c_[np.ones(n_samples), X]
            group_sizes = [1] + group_sizes
            beta_var = [cvxpy.Variable(1)] + beta_var
            num_groups = num_groups + 1
            # Compute model prediction for the intercept with no penalization
            model_prediction = (
                X[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            )
            inf_lim = 1
        group_lasso_weights_param = cvxpy.Parameter(num_groups, nonneg=True)
        for i in range(inf_lim, num_groups):
            model_prediction += (
                X[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            )
            group_lasso_penalization += (
                cvxpy.sqrt(group_sizes[i])
                * group_lasso_weights_param[i]
                * cvxpy.norm(beta_var[i], 2)
            )
        if self.quantile == False:
            objective_function = (1.0 / n_samples) * cvxpy.sum_squares(
                y - model_prediction
            )
        else:
            objective_function = (1.0 / n_samples) * cvxpy.sum(
                self._quantile_function(X=(y - model_prediction))
            )
        objective = cvxpy.Minimize(objective_function + group_lasso_penalization)
        problem = cvxpy.Problem(objective)
        beta_sol_list = []
        # Solve the problem iteratively for each parameter value
        for lam, gl in param:
            group_lasso_weights_param.value = lam * gl
            # Solve the problem. If solver is left as default, try optimal solver sugested by cvxpy.
            # If other name is provided, try the name provided
            # If these options fail, try default ECOS, OSQP, SCS options
            try:
                if self.solver == "default":
                    problem.solve(warm_start=True)
                else:
                    solver_dict = self._cvxpy_solver_options(solver=self.solver)
                    problem.solve(**solver_dict)
            except (ValueError, cvxpy.error.SolverError):
                logging.warning(
                    "Default solver failed. Using alternative options. Check solver and solver_stats for more "
                    "details"
                )
                solver = ["ECOS", "OSQP", "SCS"]
                for elt in solver:
                    solver_dict = self._cvxpy_solver_options(solver=elt)
                    try:
                        problem.solve(**solver_dict)
                        if "optimal" in problem.status:
                            break
                    except (ValueError, cvxpy.error.SolverError):
                        continue
            self.solver_stats = problem.solver_stats
            if problem.status in ["infeasible", "unbounded"]:
                logging.warning("Optimization problem status failure")
            beta_sol = np.concatenate([b.value for b in beta_var], axis=0)
            beta_sol[np.abs(beta_sol) < self.tol] = 0
            beta_sol_list.append(beta_sol)
        return beta_sol_list

    def asgl(self, X, y, group_index, param):
        """
        adaptive sparse group lasso penalized solver
        """
        n_samples, n_features = X.shape
        # Check th group_index, find the unique groups, count how many vars are in each group (this is the group size)
        group_index = np.asarray(group_index).astype(int)
        unique_group_index = np.unique(group_index)
        group_sizes, beta_var = self._num_beta_var_from_group_index(group_index)
        num_groups = len(group_sizes)
        model_prediction = 0
        alasso_penalization = 0
        a_group_lasso_penalization = 0
        # If the model has an intercept, we calculate the value of the model for the intercept group_index
        # We start the penalization in inf_lim so if the model has an intercept, penalization starts after the intercept
        inf_lim = 0
        if self.fit_intercept:
            group_index = np.append(0, group_index)
            unique_group_index = np.unique(group_index)
            X = np.c_[np.ones(n_samples), X]
            n_features = n_features + 1
            group_sizes = [1] + group_sizes
            beta_var = [cvxpy.Variable(1)] + beta_var
            num_groups = num_groups + 1
            model_prediction = (
                X[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            )
            inf_lim = 1
        l_weights_param = cvxpy.Parameter(n_features, nonneg=True)
        group_lasso_weights_param = cvxpy.Parameter(num_groups, nonneg=True)
        for i in range(inf_lim, num_groups):
            model_prediction += (
                X[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            )
            a_group_lasso_penalization += (
                cvxpy.sqrt(group_sizes[i])
                * group_lasso_weights_param[i]
                * cvxpy.norm(beta_var[i], 2)
            )
            alasso_penalization += l_weights_param[
                np.where(group_index == unique_group_index[i])[0]
            ].T @ cvxpy.abs(beta_var[i])
        if self.quantile == False:
            objective_function = (1.0 / n_samples) * cvxpy.sum_squares(
                y - model_prediction
            )
        else:
            objective_function = (1.0 / n_samples) * cvxpy.sum(
                self._quantile_function(X=(y - model_prediction))
            )
        objective = cvxpy.Minimize(
            objective_function + a_group_lasso_penalization + alasso_penalization
        )
        problem = cvxpy.Problem(objective)
        beta_sol_list = []
        # Solve the problem iteratively for each parameter value
        for lam, al, lw, glw in param:
            l_weights_param.value = lw * lam * al
            group_lasso_weights_param.value = glw * lam * (1 - al)
            # Solve the problem. If solver is left as default, try optimal solver sugested by cvxpy.
            # If other name is provided, try the name provided
            # If these options fail, try default ECOS, OSQP, SCS options
            try:
                if self.solver == "default":
                    problem.solve(warm_start=True)
                else:
                    solver_dict = self._cvxpy_solver_options(solver=self.solver)
                    problem.solve(**solver_dict)
            except (ValueError, cvxpy.error.SolverError):
                logging.warning(
                    "Default solver failed. Using alternative options. Check solver and solver_stats for more "
                    "details"
                )
                solver = ["ECOS", "OSQP", "SCS"]
                for elt in solver:
                    solver_dict = self._cvxpy_solver_options(solver=elt)
                    try:
                        problem.solve(**solver_dict)
                        if "optimal" in problem.status:
                            break
                    except (ValueError, cvxpy.error.SolverError):
                        continue
            self.solver_stats = problem.solver_stats
            if problem.status in ["infeasible", "unbounded"]:
                logging.warning("Optimization problem status failure")
            beta_sol = np.concatenate([b.value for b in beta_var], axis=0)
            beta_sol[np.abs(beta_sol) < self.tol] = 0
            beta_sol_list.append(beta_sol)
        return beta_sol_list

    # PARALLEL CODE ###################################################################################################

    def _parallel_execution(self, X, y, param, group_index=None):
        """
        Parallel implementation of the solvers
        """
        if self.n_jobs is None:
            # If the number of cores is not defined by user, use all available minus 1
            num_chunks = mp.cpu_count() - 1
        else:
            num_chunks = np.min((self.n_jobs, mp.cpu_count()))
        # Divide the list of parameter values into as many chunks as threads that we want to use in parallel
        tmp_group_index_chunks = np.array_split(range(len(param)), num_chunks)
        # If the number of parameters is shorter than the number of threads, delete the empty groups
        group_index_chunks = []
        for elt in tmp_group_index_chunks:
            if elt.size != 0:
                group_index_chunks.append(elt)
        # chunks is a list with as many elements as num_chunks
        # Each element of the list is another list of tuples of parameter values
        chunks = []
        for elt in group_index_chunks:
            chunks.append(param[elt[0] : (1 + elt[-1])])
        # Solve problem in parallel
        pool = mp.Pool(num_chunks)
        if self.penalization in ["lasso", "alasso"]:
            global_results = pool.map(
                functools.partial(getattr(self, self._get_solver_names()), X, y), chunks
            )
        else:
            global_results = pool.map(
                functools.partial(
                    getattr(self, self._get_solver_names()), X, y, group_index
                ),
                chunks,
            )
        pool.close()
        pool.join()
        # Re-build the output of the function
        beta_sol_list = []
        if len(param) < num_chunks:
            limit = len(param)
        else:
            limit = num_chunks
        for i in range(limit):
            beta_sol_list.extend(global_results[i])
        return beta_sol_list

    # FIT METHOD ######################################################################################################

    def _get_solver_names(self):
        if "asgl" in self.penalization:
            return "asgl"
        else:
            return self.penalization

    def fit(self, X, y, group_index=None):
        """
        Main function of the module. Given a model, penalization and parameter values specified in the class definition,
        this function solves the model and produces the regression coefficients
        """
        return NotImplementedError

    # PREDICTION METHOD ###############################################################################################

    def predict(self, X):
        """
        To be executed after fitting a model. Given a new dataset, this function produces predictions for that data
        considering the different model coefficients output provided by function fit
        """
        if self.fit_intercept:
            X = np.c_[np.ones(_num_samples(X)), X]

        return np.dot(X, self.coef_)

    # NUMBER OF PARAMETERS ############################################################################################

    def _num_parameters(self):
        """
        retrieves the number of parameters to be considered in a model
        Output: tuple [num_models, n_lambda, n_alpha, n_l_weights, n_group_lasso_weights] where
        - num_models: total number of models to be solved for the grid of parameters given
        - n_lambda: number of different lambda1 values
        - n_alpha: number of different alpha values
        - n_l_weights: number of different weights for the lasso part of the asgl (or asgl_lasso) penalizations
        - n_group_lasso_weights: number of different weights for the lasso part of the asgl (or asgl_gl) penalizations
        """
        # Run the input_checker to verify that the inputs have the correct format
        if self._input_checker() is False:
            logging.error("incorrect input parameters")
            raise ValueError("incorrect input parameters")
        if self.penalization is None:
            # See meaning of each element in the "else" result statement.
            result = dict(
                num_models=1,
                n_lambda=None,
                n_alpha=None,
                n_lasso_weights=None,
                n_group_lasso_weights=None,
            )
        else:
            n_lambda, drop = self._preprocessing_lambda()
            n_alpha, drop = self._preprocessing_alpha()
            n_lasso_weights, drop = self._preprocessing_weights(self.lasso_weights)
            n_group_lasso_weights, drop = self._preprocessing_weights(
                self.group_lasso_weights
            )
            list_param = [n_lambda, n_alpha, n_lasso_weights, n_group_lasso_weights]
            list_param_no_none = [elt for elt in list_param if elt is not None]
            num_models = np.prod(list_param_no_none)
            result = dict(
                num_models=num_models,
                n_lambda=n_lambda,
                n_alpha=n_alpha,
                n_lasso_weights=n_lasso_weights,
                n_group_lasso_weights=n_group_lasso_weights,
            )
        return result

    def _retrieve_parameters_idx(self, param_index):
        """
        Given an index for the param iterable output from _preprocessing function, this function returns a tupple
        with the index of the value of each parameter.
        Example: Solving an adaptive sparse group lasso model with 5 values for lambda1, 4 values for alpha,
                 3 possible lasso weights and 3 possible group lasso weights yields in a grid search on
                 5*4*3*3=180 parameters.
                 Inputing param_index=120 (out of the 180 possible values)in this function will output the
                 lambda, alpha, and weights index for such value
        If the penalization under consideration does not include any of the required parameters (for example, if we are
        solving an sparse group lasso model, we do not consider adaptive weights), the output regarding the non used
        parameters are set to be None.
        """
        number_parameters = self._num_parameters()
        n_models, n_lambda, n_alpha, n_l_weights, n_group_lasso_weights = [
            number_parameters[elt] for elt in number_parameters
        ]
        if param_index > n_models:
            string = (
                f"param_index should be smaller or equal than the number of models solved. n_models={n_models}, "
                f"param_index={param_index}"
            )
            logging.error(string)
            raise ValueError(string)
        # If penalization is None, all parameters are set to None
        if self.penalization is None:
            result = [None, None, None, None]
        # If penalization is lasso or gl, there is only one parameter, so param_index = position of that parameter
        elif self.penalization in ["lasso", "gl"]:
            result = [param_index, None, None, None]
        # If penalization is sgl, there are two parameters and two None
        elif self.penalization == "sgl":
            parameter_matrix = np.arange(n_models).reshape((n_lambda, n_alpha))
            parameter_idx = np.where(parameter_matrix == param_index)
            result = [parameter_idx[0][0], parameter_idx[1][0], None, None]
        elif self.penalization == "alasso":
            parameter_matrix = np.arange(n_models).reshape((n_lambda, n_l_weights))
            parameter_idx = np.where(parameter_matrix == param_index)
            result = [parameter_idx[0][0], None, parameter_idx[1][0], None]
        elif self.penalization == "agl":
            parameter_matrix = np.arange(n_models).reshape(
                (n_lambda, n_group_lasso_weights)
            )
            parameter_idx = np.where(parameter_matrix == param_index)
            result = [parameter_idx[0][0], None, None, parameter_idx[1][0]]
        else:
            parameter_matrix = np.arange(n_models).reshape(
                (n_lambda, n_alpha, n_l_weights, n_group_lasso_weights)
            )
            parameter_idx = np.where(parameter_matrix == param_index)
            result = [
                parameter_idx[0][0],
                parameter_idx[1][0],
                parameter_idx[2][0],
                parameter_idx[3][0],
            ]
        return result

    def retrieve_parameters_value(self, param_index):
        """
        Converts the index output from retrieve_parameters_idx into the actual numerical value of the parameters.
        Outputs None if the parameter was not used in the model formulation.
        To be executed after fit method.
        """
        param_index = self._retrieve_parameters_idx(param_index)
        result = [
            param[idx] if idx is not None else None
            for idx, param in zip(
                param_index,
                [
                    self.lambda1,
                    self.alpha,
                    self.lasso_weights,
                    self.group_lasso_weights,
                ],
            )
        ]
        result = dict(
            lambda1=result[0],
            alpha=result[1],
            lasso_weights=result[2],
            group_lasso_weights=result[3],
        )
        return result


class LinearRegression(BaseModel):
    def __init__(
        self,
        alpha=1.0,
        *,
        penalization=None,
        quantile=False,
        adaptive=False,
        group=False,
        sparse=False,
        fit_intercept=True,
        tol=1e-5,
        lambda1=1.0,
        tau=0.5,
        lasso_weights=None,
        group_lasso_weights=None,
        n_jobs=None,
        solver="default",
        max_iter=500,
    ):
        super().__init__(
            alpha=alpha,
            penalization=penalization,
            quantile=quantile,
            adaptive=adaptive,
            group=group,
            sparse=sparse,
            fit_intercept=fit_intercept,
            tol=tol,
            lambda1=lambda1,
            tau=tau,
            lasso_weights=lasso_weights,
            group_lasso_weights=group_lasso_weights,
            n_jobs=n_jobs,
            solver=solver,
            max_iter=max_iter,
        )

    def fit(self, X, y):
        n_samples = _num_samples(X)
        n_features = _num_features(X)
        # If we want an fit_intercept, it adds a column of ones to the matrix X
        if self.fit_intercept:
            n_features = n_features + 1
            X = np.c_[np.ones(n_samples), X]
        # Define the objective function
        beta_var = cvxpy.Variable(n_features)
        if self.quantile == False:
            objective_function = (1.0 / n_samples) * cvxpy.sum_squares(y - X @ beta_var)
        else:
            objective_function = (1.0 / n_samples) * cvxpy.sum(
                self._quantile_function(X=(y - X @ beta_var))
            )
        objective = cvxpy.Minimize(objective_function)
        problem = cvxpy.Problem(objective)
        # Solve the problem. If solver is left as default, try optimal solver sugested by cvxpy.
        # If other name is provided, try the name provided
        # If these options fail, try default ECOS, OSQP, SCS options
        try:
            if self.solver == "default":
                problem.solve(warm_start=True)
            else:
                solver_dict = self._cvxpy_solver_options(solver=self.solver)
                problem.solve(**solver_dict)
        except (ValueError, cvxpy.error.SolverError):
            logging.warning(
                "Default solver failed. Using alternative options. Check solver and solver_stats for more "
                "details"
            )
            solver = ["ECOS", "OSQP", "SCS"]
            for elt in solver:
                solver_dict = self._cvxpy_solver_options(solver=elt)
                try:
                    problem.solve(**solver_dict)
                    if "optimal" in problem.status:
                        break
                except (ValueError, cvxpy.error.SolverError):
                    continue
        self.solver_stats = problem.solver_stats
        if problem.status in ["infeasible", "unbounded"]:
            logging.warning("Optimization problem status failure")
        self.coef_ = beta_var.value
        self.coef_[np.abs(self.coef_) < self.tol] = 0
        return self


class QuantileRegression(LinearRegression):
    def __init__(
        self,
        alpha=1.0,
        *,
        penalization=None,
        quantile=True,
        adaptive=False,
        group=False,
        sparse=False,
        fit_intercept=True,
        tol=1e-5,
        lambda1=1.0,
        tau=0.5,
        lasso_weights=None,
        group_lasso_weights=None,
        n_jobs=None,
        solver="default",
        max_iter=500,
    ):
        super().__init__(
            alpha=alpha,
            penalization=penalization,
            quantile=quantile,
            adaptive=adaptive,
            group=group,
            sparse=sparse,
            fit_intercept=fit_intercept,
            tol=tol,
            lambda1=lambda1,
            tau=tau,
            lasso_weights=lasso_weights,
            group_lasso_weights=group_lasso_weights,
            n_jobs=n_jobs,
            solver=solver,
            max_iter=max_iter,
        )


class Lasso(BaseModel):
    def __init__(
        self,
        alpha=1.0,
        *,
        penalization="L1",
        quantile=False,
        adaptive=False,
        group=False,
        sparse=False,
        fit_intercept=True,
        tol=1e-5,
        lambda1=1.0,
        tau=0.5,
        lasso_weights=None,
        group_lasso_weights=None,
        n_jobs=None,
        solver="default",
        max_iter=500,
    ):
        super().__init__(
            alpha=alpha,
            penalization=penalization,
            quantile=quantile,
            adaptive=adaptive,
            group=group,
            sparse=sparse,
            fit_intercept=fit_intercept,
            tol=tol,
            lambda1=lambda1,
            tau=tau,
            lasso_weights=lasso_weights,
            group_lasso_weights=group_lasso_weights,
            n_jobs=n_jobs,
            solver=solver,
            max_iter=max_iter,
        )

    def fit(self, X, y, *, group_index=None):
        """
        Lasso penalized solver
        """
        # n_samples, n_features = X.shape
        n_samples = _num_samples(X)
        n_features = _num_features(X)
        # If we want an intercept, it adds a column of ones to the matrix X.
        # Init_pen controls when the penalization starts, this way the intercept is not penalized
        if self.fit_intercept:
            n_features = n_features + 1
            X = np.c_[np.ones(n_samples), X]
            init_pen = 1
        else:
            init_pen = 0
        # Define the objective function
        lambda_param = cvxpy.Parameter(nonneg=True)
        beta_var = cvxpy.Variable(n_features)
        lasso_penalization = lambda_param * cvxpy.norm(beta_var[init_pen:], 1)
        if self.quantile == False:
            objective_function = (1.0 / n_samples) * cvxpy.sum_squares(y - X @ beta_var)
        else:
            objective_function = (1.0 / n_samples) * cvxpy.sum(
                self._quantile_function(X=(y - X @ beta_var))
            )
        objective = cvxpy.Minimize(objective_function + lasso_penalization)
        problem = cvxpy.Problem(objective)

        # Solve the problem iteratively for each parameter value
        lambda_param.value = self.lambda1

        # Solve the problem. If solver is left as default, try optimal solver sugested by cvxpy.
        # If other name is provided, try the name provided
        # If these options fail, try default ECOS, OSQP, SCS options
        try:
            if self.solver == "default":
                problem.solve(warm_start=True)
            else:
                solver_dict = self._cvxpy_solver_options(solver=self.solver)
                problem.solve(**solver_dict)
        except (ValueError, cvxpy.error.SolverError):
            logging.warning(
                "Default solver failed. Using alternative options. Check solver and solver_stats for more "
                "details"
            )
            solver = ["ECOS", "OSQP", "SCS"]
            for elt in solver:
                solver_dict = self._cvxpy_solver_options(solver=elt)
                try:
                    problem.solve(**solver_dict)
                    if "optimal" in problem.status:
                        break
                except (ValueError, cvxpy.error.SolverError):
                    continue
        self.solver_stats = problem.solver_stats
        if problem.status in ["infeasible", "unbounded"]:
            logging.warning("Optimization problem status failure")

        self.coef_ = beta_var.value
        self.coef_[np.abs(self.coef_) < self.tol] = 0

        return self


class QuantileLasso(Lasso):
    def __init__(
        self,
        alpha=1.0,
        *,
        penalization="L1",
        quantile=True,
        adaptive=False,
        group=False,
        sparse=False,
        fit_intercept=True,
        tol=1e-5,
        lambda1=1.0,
        tau=0.5,
        lasso_weights=None,
        group_lasso_weights=None,
        n_jobs=None,
        solver="default",
        max_iter=500,
    ):
        super().__init__(
            alpha=alpha,
            penalization=penalization,
            quantile=quantile,
            adaptive=adaptive,
            group=group,
            sparse=sparse,
            fit_intercept=fit_intercept,
            tol=tol,
            lambda1=lambda1,
            tau=tau,
            lasso_weights=lasso_weights,
            group_lasso_weights=group_lasso_weights,
            n_jobs=n_jobs,
            solver=solver,
            max_iter=max_iter,
        )


class AdaptiveLasso(BaseModel):
    def __init__(
        self,
        alpha=1.0,
        *,
        penalization="L1",
        quantile=False,
        adaptive=True,
        group=False,
        sparse=False,
        fit_intercept=True,
        tol=1e-5,
        lambda1=1.0,
        tau=0.5,
        lasso_weights=None,
        group_lasso_weights=None,
        n_jobs=None,
        solver="default",
        max_iter=500,
    ):
        super().__init__(
            alpha=alpha,
            penalization=penalization,
            quantile=quantile,
            adaptive=adaptive,
            group=group,
            sparse=sparse,
            fit_intercept=fit_intercept,
            tol=tol,
            lambda1=lambda1,
            tau=tau,
            lasso_weights=lasso_weights,
            group_lasso_weights=group_lasso_weights,
            n_jobs=n_jobs,
            solver=solver,
            max_iter=max_iter,
        )

    def fit(self, X, y, *, group_index=None):
        """
        Lasso penalized solver
        """
        n_samples = _num_samples(X)
        n_features = _num_features(X)
        # If we want an intercept, it adds a column of ones to the matrix X.
        # Init_pen controls when the penalization starts, this way the intercept is not penalized

        if self.fit_intercept:
            n_features = n_features + 1
            X = np.c_[np.ones(n_samples), X]
            init_pen = 1
        else:
            init_pen = 0

        # Define the objective function
        l_weights_param = cvxpy.Parameter(n_features, nonneg=True)
        beta_var = cvxpy.Variable(n_features)
        lasso_penalization = cvxpy.norm(
            l_weights_param[init_pen:].T @ cvxpy.abs(beta_var[init_pen:]), 1
        )

        if self.quantile == False:
            objective_function = (1.0 / n_samples) * cvxpy.sum_squares(y - X @ beta_var)

        else:
            objective_function = (1.0 / n_samples) * cvxpy.sum(
                self._quantile_function(X=(y - X @ beta_var))
            )

        objective = cvxpy.Minimize(objective_function + lasso_penalization)
        problem = cvxpy.Problem(objective)

        # Solve the problem iteratively for each parameter value
        l_weights_param.value = np.array([self.lambda1] * _num_features(X)) * self.alpha

        # Solve the problem. If solver is left as default, try optimal solver sugested by cvxpy.
        # If other name is provided, try the name provided
        # If these options fail, try default ECOS, OSQP, SCS options

        try:
            if self.solver == "default":
                problem.solve(warm_start=True)
            else:
                solver_dict = self._cvxpy_solver_options(solver=self.solver)
                problem.solve(**solver_dict)

        except (ValueError, cvxpy.error.SolverError):
            logging.warning(
                "Default solver failed. Using alternative options. Check solver and solver_stats for more "
                "details"
            )
            solver = ["ECOS", "OSQP", "SCS"]
            for elt in solver:
                solver_dict = self._cvxpy_solver_options(solver=elt)
                try:
                    problem.solve(**solver_dict)
                    if "optimal" in problem.status:
                        break
                except (ValueError, cvxpy.error.SolverError):
                    continue

        # self.solver_stats = problem.solver_stats
        if problem.status in ["infeasible", "unbounded"]:
            logging.warning("Optimization problem status failure")

        self.coef_ = beta_var.value
        self.coef_[np.abs(self.coef_) < self.tol] = 0
        print(self.coef_)

        return self


class QuantileAdaptiveLasso(AdaptiveLasso):
    def __init__(
        self,
        alpha=1.0,
        *,
        penalization="L1",
        quantile=True,
        adaptive=True,
        group=False,
        sparse=False,
        fit_intercept=True,
        tol=1e-5,
        lambda1=1.0,
        tau=0.5,
        lasso_weights=None,
        group_lasso_weights=None,
        n_jobs=None,
        solver="default",
        max_iter=500,
    ):
        super().__init__(
            alpha=alpha,
            penalization=penalization,
            quantile=quantile,
            adaptive=adaptive,
            group=group,
            sparse=sparse,
            fit_intercept=fit_intercept,
            tol=tol,
            lambda1=lambda1,
            tau=tau,
            lasso_weights=lasso_weights,
            group_lasso_weights=group_lasso_weights,
            n_jobs=n_jobs,
            solver=solver,
            max_iter=max_iter,
        )


class GroupLasso(BaseModel):
    def __init__(
        self,
        alpha=1.0,
        *,
        penalization="L1",
        quantile=False,
        adaptive=False,
        group=True,
        sparse=False,
        fit_intercept=True,
        tol=1e-5,
        lambda1=1.0,
        tau=0.5,
        lasso_weights=None,
        group_lasso_weights=None,
        n_jobs=None,
        solver="default",
        max_iter=500,
    ):
        super().__init__(
            alpha=alpha,
            penalization=penalization,
            quantile=quantile,
            adaptive=adaptive,
            group=group,
            sparse=sparse,
            fit_intercept=fit_intercept,
            tol=tol,
            lambda1=lambda1,
            tau=tau,
            lasso_weights=lasso_weights,
            group_lasso_weights=group_lasso_weights,
            n_jobs=n_jobs,
            solver=solver,
            max_iter=max_iter,
        )

    def fit(self, X, y, *, group_index=None):
        """
        Group lasso penalized solver
        """
        n_samples = _num_samples(X)
        n_features = _num_features(X)

        if group_index is None:
            group_index = np.ones(n_features)

        # Check th group_index, find the unique groups, count how many vars are in each group (this is the group size)
        group_index = np.asarray(group_index).astype(int)
        if (group_index == 0).any():
            group_index += 1

        unique_group_index = np.unique(group_index)
        group_sizes, beta_var = self._num_beta_var_from_group_index(group_index)
        num_groups = len(group_sizes)
        model_prediction = 0
        group_lasso_penalization = 0
        # If the model has an intercept, we calculate the value of the model for the intercept group_index
        # We start the penalization in inf_lim so if the model has an intercept, penalization starts after the intercept
        inf_lim = 0
        if self.fit_intercept:
            # Adds an element (referring to the intercept) to group_index, group_sizes, num groups
            group_index = np.append(0, group_index)
            unique_group_index = np.unique(group_index)
            X = np.c_[np.ones(n_samples), X]
            group_sizes = [1] + group_sizes
            beta_var = [cvxpy.Variable(1)] + beta_var
            num_groups = num_groups + 1
            # Compute model prediction for the intercept with no penalization
            model_prediction = (
                X[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            )
            inf_lim = 1
        for i in range(inf_lim, num_groups):
            model_prediction += (
                X[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            )
            group_lasso_penalization += cvxpy.sqrt(group_sizes[i]) * cvxpy.norm(
                beta_var[i], 2
            )
        if self.quantile == False:
            objective_function = (1.0 / n_samples) * cvxpy.sum_squares(
                y - model_prediction
            )
        else:
            objective_function = (1.0 / n_samples) * cvxpy.sum(
                self._quantile_function(X=(y - model_prediction))
            )
        lambda_param = cvxpy.Parameter(nonneg=True)
        objective = cvxpy.Minimize(
            objective_function + (lambda_param * group_lasso_penalization)
        )
        problem = cvxpy.Problem(objective)
        beta_sol_list = []
        # Solve the problem iteratively for each parameter value
        lambda_param.value = self.lambda1
        # Solve the problem. If solver is left as default, try optimal solver sugested by cvxpy.
        # If other name is provided, try the name provided
        # If these options fail, try default ECOS, OSQP, SCS options
        try:
            if self.solver == "default":
                problem.solve(warm_start=True)
            else:
                solver_dict = self._cvxpy_solver_options(solver=self.solver)
                problem.solve(**solver_dict)
        except (ValueError, cvxpy.error.SolverError):
            logging.warning(
                "Default solver failed. Using alternative options. Check solver and solver_stats for more "
                "details"
            )
            solver = ["ECOS", "OSQP", "SCS"]
            for elt in solver:
                solver_dict = self._cvxpy_solver_options(solver=elt)
                try:
                    problem.solve(**solver_dict)
                    if "optimal" in problem.status:
                        break
                except (ValueError, cvxpy.error.SolverError):
                    continue

        self.solver_stats = problem.solver_stats
        if problem.status in ["infeasible", "unbounded"]:
            logging.warning("Optimization problem status failure")
        self.coef_ = np.concatenate([b.value for b in beta_var], axis=0)
        self.coef_[np.abs(self.coef_) < self.tol] = 0

        return self


class QuantileGroupLasso(GroupLasso):
    def __init__(
        self,
        alpha=1.0,
        *,
        penalization="L1",
        quantile=True,
        adaptive=False,
        group=True,
        sparse=False,
        fit_intercept=True,
        tol=1e-5,
        lambda1=1.0,
        tau=0.5,
        lasso_weights=None,
        group_lasso_weights=None,
        n_jobs=None,
        solver="default",
        max_iter=500,
    ):
        super().__init__(
            alpha=alpha,
            penalization=penalization,
            quantile=quantile,
            adaptive=adaptive,
            group=group,
            sparse=sparse,
            fit_intercept=fit_intercept,
            tol=tol,
            lambda1=lambda1,
            tau=tau,
            lasso_weights=lasso_weights,
            group_lasso_weights=group_lasso_weights,
            n_jobs=n_jobs,
            solver=solver,
            max_iter=max_iter,
        )


class SparseGroupLasso(BaseModel):
    def __init__(
        self,
        alpha=1.0,
        *,
        penalization="L1",
        quantile=False,
        adaptive=False,
        group=True,
        sparse=True,
        fit_intercept=True,
        tol=1e-5,
        lambda1=1.0,
        tau=0.5,
        lasso_weights=None,
        group_lasso_weights=None,
        n_jobs=None,
        solver="default",
        max_iter=500,
    ):
        super().__init__(
            alpha=alpha,
            penalization=penalization,
            quantile=quantile,
            adaptive=adaptive,
            group=group,
            sparse=sparse,
            fit_intercept=fit_intercept,
            tol=tol,
            lambda1=lambda1,
            tau=tau,
            lasso_weights=lasso_weights,
            group_lasso_weights=group_lasso_weights,
            n_jobs=n_jobs,
            solver=solver,
            max_iter=max_iter,
        )

    def fit(self, X, y, *, group_index=None):
        """
        Sparse group lasso penalized solver
        """
        n_samples = _num_samples(X)
        n_features = _num_features(X)

        if group_index is None:
            group_index = np.ones(n_features)

        # Check th group_index, find the unique groups, count how many vars are in each group (this is the group size)
        group_index = np.asarray(group_index).astype(int)
        if (group_index == 0).any():
            group_index += 1
        unique_group_index = np.unique(group_index)
        group_sizes, beta_var = self._num_beta_var_from_group_index(group_index)
        num_groups = len(group_sizes)
        model_prediction = 0
        lasso_penalization = 0
        group_lasso_penalization = 0
        # If the model has an intercept, we calculate the value of the model for the intercept group_index
        # We start the penalization in inf_lim so if the model has an intercept, penalization starts after the intercept
        inf_lim = 0
        if self.fit_intercept:
            group_index = np.append(0, group_index)
            unique_group_index = np.unique(group_index)
            X = np.c_[np.ones(n_samples), X]
            group_sizes = [1] + group_sizes
            beta_var = [cvxpy.Variable(1)] + beta_var
            num_groups = num_groups + 1
            model_prediction = (
                X[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            )
            inf_lim = 1
        for i in range(inf_lim, num_groups):
            model_prediction += (
                X[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            )
            group_lasso_penalization += cvxpy.sqrt(group_sizes[i]) * cvxpy.norm(
                beta_var[i], 2
            )
            lasso_penalization += cvxpy.norm(beta_var[i], 1)
        if self.quantile == False:
            objective_function = (1.0 / n_samples) * cvxpy.sum_squares(
                y - model_prediction
            )
        else:
            objective_function = (1.0 / n_samples) * cvxpy.sum(
                self._quantile_function(X=(y - model_prediction))
            )
        lasso_param = cvxpy.Parameter(nonneg=True)
        grp_lasso_param = cvxpy.Parameter(nonneg=True)
        objective = cvxpy.Minimize(
            objective_function
            + (grp_lasso_param * group_lasso_penalization)
            + (lasso_param * lasso_penalization)
        )
        problem = cvxpy.Problem(objective)
        # Solve the problem iteratively for each parameter value

        lasso_param.value = self.lambda1 * self.alpha
        grp_lasso_param.value = self.lambda1 * (1 - self.alpha)
        # Solve the problem. If solver is left as default, try optimal solver sugested by cvxpy.
        # If other name is provided, try the name provided
        # If these options fail, try default ECOS, OSQP, SCS options
        try:
            if self.solver == "default":
                problem.solve(warm_start=True)
            else:
                solver_dict = self._cvxpy_solver_options(solver=self.solver)
                problem.solve(**solver_dict)
        except (ValueError, cvxpy.error.SolverError):
            logging.warning(
                "Default solver failed. Using alternative options. Check solver and solver_stats for more "
                "details"
            )
            solver = ["ECOS", "OSQP", "SCS"]
            for elt in solver:
                solver_dict = self._cvxpy_solver_options(solver=elt)
                try:
                    problem.solve(**solver_dict)
                    if "optimal" in problem.status:
                        break
                except (ValueError, cvxpy.error.SolverError):
                    continue
        self.solver_stats = problem.solver_stats
        if problem.status in ["infeasible", "unbounded"]:
            logging.warning("Optimization problem status failure")
        self.coef_ = np.concatenate([b.value for b in beta_var], axis=0)
        self.coef_[np.abs(self.coef_) < self.tol] = 0

        return self


class QuantileSparseGroupLasso(SparseGroupLasso):
    def __init__(
        self,
        alpha=1.0,
        *,
        penalization="L1",
        quantile=True,
        adaptive=False,
        group=True,
        sparse=True,
        fit_intercept=True,
        tol=1e-5,
        lambda1=1.0,
        tau=0.5,
        lasso_weights=None,
        group_lasso_weights=None,
        n_jobs=None,
        solver="default",
        max_iter=500,
    ):
        super().__init__(
            alpha=alpha,
            penalization=penalization,
            quantile=quantile,
            adaptive=adaptive,
            group=group,
            sparse=sparse,
            fit_intercept=fit_intercept,
            tol=tol,
            lambda1=lambda1,
            tau=tau,
            lasso_weights=lasso_weights,
            group_lasso_weights=group_lasso_weights,
            n_jobs=n_jobs,
            solver=solver,
            max_iter=max_iter,
        )


class AdaptiveSparseGroupLasso(BaseModel):
    def __init__(
        self,
        alpha=1.0,
        *,
        penalization="L1",
        quantile=False,
        adaptive=True,
        group=True,
        sparse=True,
        fit_intercept=True,
        tol=1e-5,
        lambda1=1.0,
        tau=0.5,
        lasso_weights=None,
        group_lasso_weights=None,
        n_jobs=None,
        solver="default",
        max_iter=500,
    ):
        super().__init__(
            alpha=alpha,
            penalization=penalization,
            quantile=quantile,
            adaptive=adaptive,
            group=group,
            sparse=sparse,
            fit_intercept=fit_intercept,
            tol=tol,
            lambda1=lambda1,
            tau=tau,
            lasso_weights=lasso_weights,
            group_lasso_weights=group_lasso_weights,
            n_jobs=n_jobs,
            solver=solver,
            max_iter=max_iter,
        )

    def fit(self, X, y, *, group_index=None):
        """
        adaptive sparse group lasso penalized solver
        """
        n_samples = _num_samples(X)
        n_features = _num_features(X)

        if group_index is None:
            group_index = np.ones(n_features)

        if self.lasso_weights is None:
            lasso_weights = np.array([self.alpha] * (n_features + 1)).reshape(
                -1,
            )

        print()
        # Check th group_index, find the unique groups, count how many vars are in each group (this is the group size)
        group_index = np.asarray(group_index).astype(int)
        unique_group_index = np.unique(group_index)
        group_sizes, beta_var = self._num_beta_var_from_group_index(group_index)
        num_groups = len(group_sizes)

        if self.group_lasso_weights is None:
            group_lasso_weights = np.array([self.alpha] * (num_groups + 1)).reshape(
                -1,
            )

        model_prediction = 0
        alasso_penalization = 0
        a_group_lasso_penalization = 0
        # If the model has an intercept, we calculate the value of the model for the intercept group_index
        # We start the penalization in inf_lim so if the model has an intercept, penalization starts after the intercept
        inf_lim = 0
        if self.fit_intercept:
            group_index = np.append(0, group_index)
            unique_group_index = np.unique(group_index)
            X = np.c_[np.ones(n_samples), X]
            n_features = n_features + 1
            group_sizes = [1] + group_sizes
            beta_var = [cvxpy.Variable(1)] + beta_var
            num_groups = num_groups + 1
            model_prediction = (
                X[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            )
            inf_lim = 1
        l_weights_param = cvxpy.Parameter(n_features, nonneg=True)
        group_lasso_weights_param = cvxpy.Parameter(num_groups, nonneg=True)
        for i in range(inf_lim, num_groups):
            model_prediction += (
                X[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            )
            a_group_lasso_penalization += (
                cvxpy.sqrt(group_sizes[i])
                * group_lasso_weights_param[i]
                * cvxpy.norm(beta_var[i], 2)
            )
            alasso_penalization += l_weights_param[
                np.where(group_index == unique_group_index[i])[0]
            ].T @ cvxpy.abs(beta_var[i])
        if self.quantile == False:
            objective_function = (1.0 / n_samples) * cvxpy.sum_squares(
                y - model_prediction
            )
        else:
            objective_function = (1.0 / n_samples) * cvxpy.sum(
                self._quantile_function(X=(y - model_prediction))
            )
        objective = cvxpy.Minimize(
            objective_function + a_group_lasso_penalization + alasso_penalization
        )
        problem = cvxpy.Problem(objective)
        beta_sol_list = []
        # Solve the problem iteratively for each parameter value

        print(lasso_weights.shape)
        print(group_lasso_weights.shape)
        print(lasso_weights * self.lambda1 * self.alpha)
        print((group_lasso_weights * self.lambda1 * (1 - self.alpha)))

        l_weights_param.value = lasso_weights * self.lambda1 * self.alpha
        group_lasso_weights_param.value = (
            group_lasso_weights * self.lambda1 * (1 - self.alpha)
        )
        # Solve the problem. If solver is left as default, try optimal solver sugested by cvxpy.
        # If other name is provided, try the name provided
        # If these options fail, try default ECOS, OSQP, SCS options
        try:
            if self.solver == "default":
                problem.solve(warm_start=True)
            else:
                solver_dict = self._cvxpy_solver_options(solver=self.solver)
                problem.solve(**solver_dict)
        except (ValueError, cvxpy.error.SolverError):
            logging.warning(
                "Default solver failed. Using alternative options. Check solver and solver_stats for more "
                "details"
            )
            solver = ["ECOS", "OSQP", "SCS"]
            for elt in solver:
                solver_dict = self._cvxpy_solver_options(solver=elt)
                try:
                    problem.solve(**solver_dict)
                    if "optimal" in problem.status:
                        break
                except (ValueError, cvxpy.error.SolverError):
                    continue
        self.solver_stats = problem.solver_stats
        if problem.status in ["infeasible", "unbounded"]:
            logging.warning("Optimization problem status failure")
        self.coef_ = np.concatenate([b.value for b in beta_var], axis=0)
        self.coef_[np.abs(self.coef_) < self.tol] = 0

        return self


class QuantileAdaptiveSparseGroupLasso(AdaptiveSparseGroupLasso):
    def __init__(
        self,
        alpha=1.0,
        *,
        penalization="L1",
        quantile=True,
        adaptive=True,
        group=True,
        sparse=True,
        fit_intercept=True,
        tol=1e-5,
        lambda1=1.0,
        tau=0.5,
        lasso_weights=None,
        group_lasso_weights=None,
        n_jobs=None,
        solver="default",
        max_iter=500,
    ):
        super().__init__(
            alpha=alpha,
            penalization=penalization,
            quantile=quantile,
            adaptive=adaptive,
            group=group,
            sparse=sparse,
            fit_intercept=fit_intercept,
            tol=tol,
            lambda1=lambda1,
            tau=tau,
            lasso_weights=lasso_weights,
            group_lasso_weights=group_lasso_weights,
            n_jobs=n_jobs,
            solver=solver,
            max_iter=max_iter,
        )


def _quantile_function(y_true, y_pred, tau):
    """
    Quantile function required for error computation
    """
    return (1.0 / len(y_true)) * np.sum(
        0.5 * np.abs(y_true - y_pred) + (tau - 0.5) * (y_true - y_pred)
    )


def error_calculator(y_true, prediction_list, error_type="MSE", tau=None):
    """
    Computes the error between the predicted value and the true value of the response variable
    """
    error_dict = dict(
        MSE=mean_squared_error,
        MAE=mean_absolute_error,
        MDAE=median_absolute_error,
        QRE=_quantile_function,
    )
    valid_error_types = error_dict.keys()
    # Check that the error_type is a valid error type considered
    if error_type not in valid_error_types:
        raise ValueError(
            f"invalid error type. Valid error types are {error_dict.keys()}"
        )
    if y_true.shape[0] != len(prediction_list[0]):
        logging.error("Dimension of test data does not match dimension of prediction")
        raise ValueError(
            "Dimension of test data does not match dimension of prediction"
        )
    # For each prediction, store the error associated to that prediction in a list
    error_list = []
    if error_type == "QRE":
        for y_pred in prediction_list:
            error_list.append(
                error_dict[error_type](y_true=y_true, y_pred=y_pred, tau=tau)
            )
    else:
        for y_pred in prediction_list:
            error_list.append(error_dict[error_type](y_true=y_true, y_pred=y_pred))
    return error_list
