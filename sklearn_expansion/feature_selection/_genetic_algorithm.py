# Authors: Gilles Louppe, Mathieu Blondel, Maheshakya Wijewardena
# Editor : Yuki Horie(2021)
# License: BSD 3 clause

# Ref.1 https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/cem.3226


import copy
import traceback
import pprint
import random
import importlib

import boruta
from boruta import BorutaPy

import deap
from deap import base
from deap import creator
from deap import tools

import numbers
import numpy as np
import optuna
import pandas as pd
from scipy.stats import f, chi2
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import model_selection

from sklearn.feature_selection import SelectorMixin
from sklearn.base import BaseEstimator, clone, MetaEstimatorMixin

from sklearn.exceptions import NotFittedError
from sklearn.utils.metaestimators import if_delegate_has_method


def GA(estimator, X, y,
        min_features=2,
        n_pop=150,
        n_gen=150,
        p_crossover=0.5,
        p_mutation=0.2,
        thres_selection=0.5,
        cv=model_selection.KFold(n_splits=5, shuffle=True),
        verbose=True):

    if estimator.__class__.__name__ == 'PLSRegression':
        print('n_components : ',estimator.n_components)
        min_features = estimator.n_components
        min_features = min(min_features, estimator.n_components)
    else:
        min_features = 1


    creator.create('FitnessMax', base.Fitness, weights=(1.0,))  # for minimization, set weights as (-1.0,)
    creator.create('Individual', list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    min_boundary = np.zeros(X.shape[1])
    max_boundary = np.ones(X.shape[1]) * 1.0

    def create_ind_uniform(min_boundary, max_boundary):
        index = []
        for min, max in zip(min_boundary, max_boundary):
            index.append(random.uniform(min, max))
        return index

    toolbox.register('create_ind', create_ind_uniform, min_boundary, max_boundary)
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.create_ind)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    def evalOneMax(individual):
        individual_array = np.array(individual)
        selected_X_variable_numbers = np.where(individual_array > thres_selection)[0]
        selected_X = X[:, selected_X_variable_numbers]

        if len(selected_X_variable_numbers) >= min_features:
            '''
            # if estimator is GridSearchCV(..) or OptunaSearchCV(..) then tuning the estimator by each generation
            #  GridSearchCV(..) or OptunaSearchCV(..) が与えられた場合は、各説明変数ごとにCVでハイパラをチューニングする
            if estimator.__class__.__name__ in ['GridSearchCV',
                                             'HalvingGridSearchCV',
                                             'RandomizedSearchCV',
                                             'HalvingRandomSearchCV',
                                             'OptunaSearchCV']:

                clone_model = clone(estimator)
                clone_model.fit(selected_X, y, cv=cv)
                estimator = clone_model.best_estimator_
                
            else:
                estimator = clone(estimator)
            '''

            y_pred = model_selection.cross_val_predict(estimator, selected_X, y, cv=cv)
            value = r2_score(y, y_pred)
        
        else:
            # 
            value =  - (len(selected_X_variable_numbers) - min_features)**8

        return value,

    toolbox.register('evaluate', evalOneMax)
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
    toolbox.register('select', tools.selTournament, tournsize=3)

    # random.seed(100)
    random.seed()
    pop = toolbox.population(n=n_pop)
    if verbose:
        print('Start of evolution')

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    if verbose:
        print('  Evaluated %i individuals' % len(pop))

    for generation in range(n_gen):
        if verbose:
            print('-- Generation {0} --'.format(generation + 1))

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < p_crossover:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < p_mutation:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        if verbose:
            print('  Evaluated %i individuals' % len(invalid_ind))

        pop[:] = offspring
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        if verbose:
            print('  Min %s' % min(fits))
            print('  Max %s' % max(fits))
            print('  Avg %s' % mean)
            print('  Std %s' % std)
    if verbose:
        print('-- End of (successful) evolution --')

    best_individual = tools.selBest(pop, 1)[0]
    best_individual_array = np.array(best_individual)
    selected_X_variable_numbers = np.where(best_individual_array > thres_selection)[0]

    return selected_X_variable_numbers


def _get_selected(scores, threshold, min_features, max_features):
    if max_features is not None:
        if max_features < min_features:
            print('max_features {} <  min_features {}'.format(max_features, min_features))
            print('set max_features as min_features.')
            time.sleep(0.5)
            max_features = min_features
        
    if max_features is not None:
        selected = np.zeros_like(scores, dtype=bool)
        candidate_indices = \
            np.argsort(-scores, kind='mergesort')[:max_features]

        selected[candidate_indices] = True

    else:
        selected = np.ones_like(scores, dtype=bool)
    index = scores < threshold 
    selected[index] = False

    if min_features is not None:
        candidate_indices = \
            np.argsort(-scores, kind='mergesort')[:min_features]
        selected[candidate_indices] = True
    
    return selected


class SelectByGACV(MetaEstimatorMixin, SelectorMixin, BaseEstimator):
    """Meta-transformer for selecting features based on importance weights.

    .. versionadded:: 0.17

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
        This can be both a fitted (if ``prefit`` is set to True)
        or a non-fitted estimator. The estimator must have either a
        ``feature_importances_`` or ``coef_`` attribute after fitting.

    prefit : bool, default False
        Whether a prefit model is expected to be passed into the constructor
        directly or not. If True, ``transform`` must be called directly
        and SelectFromModel cannot be used with ``cross_val_score``,
        ``GridSearchCV`` and similar utilities that clone the estimator.
        Otherwise train the model using ``fit`` and then ``transform`` to do
        feature selection.

    max_features : int or None, optional
        The maximum number of features selected scoring above ``threshold``.
        To disable ``threshold`` and only select based on ``max_features``,
        set ``threshold=-np.inf``.

    min_features : int or None, optional
        The minimum number of features selected scoring above ``threshold``.

    Attributes
    ----------
    estimator_ : an estimator
        The base estimator from which the transformer is built.
        This is stored only when a non-fitted estimator is passed to the
        ``SelectFromModel``, i.e when prefit is False.


    Notes
    -----
    Allows NaN/Inf in the input if the underlying estimator does as well.

    Examples
    --------
    >>> from sklearn_expanz.feature_selection import SelectFromGACV
    >>> from sklearn.linear_model import LogisticRegression
    >>> X = [[ 0.87, -1.34,  0.31 ],
    ...      [-2.79, -0.02, -0.85 ],
    ...      [-1.34, -0.48, -2.55 ],
    ...      [ 1.92,  1.48,  0.65 ]]
    >>> y = [0, 1, 0, 1]
    >>> selector = SelectFromGACV(estimator=LogisticRegression()).fit(X, y)
    >>> selector.get_support()
    array([False,  True, False])
    >>> selector.transform(X)
    array([[-1.34],
           [-0.02],
           [-0.48],
           [ 1.48]])
    """

    def __init__(self, estimator, 
                 n_pop=30,
                 n_gen=50,
                 p_crossover=0.5,
                 p_mutation=0.2,
                 thres_selection=0.5,
                 cv=model_selection.KFold(n_splits=5, shuffle=True),
                 scores=None,
                 threshold=None,
                 verbose=True,
                 random_state=None,
                 prefit=False,
                 max_features=None,
                 min_features=2, 
                 ):

        self.estimator = estimator
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.thres_selection = thres_selection
        self.verbose = verbose
        self.random_state=random_state
        self.cv=cv
        self.scores=scores
        self.threshold=threshold
        self.prefit = prefit
        self.max_features = max_features
        self.min_features = min_features

    def _get_support_mask(self):
        # SelectByModel can directly call on transform.
        '''    
        if self.prefit:
            estimator = self.estimator
        elif hasattr(self, 'estimator_'):
            estimator = self.estimator_
        else:
            raise ValueError('Either fit the model before transform or set'
                             ' "prefit=True" while passing the fitted'
                             ' estimator to the constructor.')
        '''
        mask = _get_selected(self.scores, self.threshold, self.min_features, self.max_features)

        if self.prefit:
            #print(mask)
            pass

        if not mask.any():
            warn(
                "No features were selected: either the data is"
                " too noisy or the selection test too strict.",
                UserWarning,
            )
            return np.empty(0).reshape((X.shape[0], 0))
        if len(mask) != self.X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")

        return mask

    def fit(self, X, y=None, **fit_params):
        """Fit the SelectFromModel meta-transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        **fit_params : Other estimator specific parameters

        Returns
        -------
        self : object
        """

        self.X = X
        self.y = y
        if self.max_features is not None:
            if not isinstance(self.max_features, numbers.Integral):
                raise TypeError("'max_features' should be an integer between"
                                " 0 and {} features. Got {!r} instead."
                                .format(X.shape[1], self.max_features))
            elif self.max_features < 0 or self.max_features > X.shape[1]:
                raise ValueError("'max_features' should be 0 and {} features."
                                 "Got {} instead."
                                 .format(X.shape[1], self.max_features))

        if self.prefit:
            if (self.scores is not None) and (self.threshold is not None):
                # skip re-feature_selection            
                return self
            else:
                raise ValueError('"prefit = True" is only applicable if both scores and threshold has already been calculated ')

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)

        index = GA(self.estimator, self.X, self.y,
                    min_features=self.min_features,
                    n_pop=self.n_pop,
                    n_gen=self.n_gen,
                    p_crossover=self.p_crossover,
                    p_mutation=self.p_mutation,
                    thres_selection=self.thres_selection,
                    cv=self.cv,
                    verbose=self.verbose)

        support_ = np.zeros_like(self.X[0], dtype=bool)
        support_[index] = True

        scores = np.zeros_like(self.X[0])
        scores[index] = 1
        
        threshold = 0.5

        self.support_ = support_
        self.scores = scores
        self.threshold = threshold

        return self


    def _more_tags(self):
        estimator_tags = self.estimator._get_tags()
        return {'allow_nan': estimator_tags.get('allow_nan', True)}
