from sklearn.tree import DecisionTreeRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import resample
import numpy as np
from joblib import Parallel, delayed


from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.utils import resample
from joblib import Parallel, delayed
import numpy as np
from numpy.random import RandomState

class GRPForest(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, eps=0.1, 
                 criterion='squared_error', splitter='best', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features=None, random_state=None, max_leaf_nodes=None,
                 min_impurity_decrease=0.0, ccp_alpha=0.0, n_jobs=1):
        self.n_estimators = n_estimators
        self.eps = eps
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.n_jobs = n_jobs
    
    def _fit_single_tree(self, X, y, sample_weight, random_state):
        local_rng = RandomState(random_state)        
        # Initialize GaussianRandomProjection for each tree
        target_n_components = min(X.shape[1], int(X.shape[1] * (1 - self.eps)))
        random_projection = GaussianRandomProjection(n_components=target_n_components, eps=self.eps, 
                                                     random_state=local_rng)
        
        n_samples = X.shape[0]
        if self.bootstrap:
            indices = local_rng.choice(n_samples, n_samples, replace=True)
            curr_sample_weight = np.bincount(indices, minlength=n_samples)
            curr_sample_weight = curr_sample_weight.astype(np.float64)
            if sample_weight is not None:
                curr_sample_weight *= sample_weight
            X_sample, y_sample = X[indices], y[indices]
        else:
            X_sample, y_sample, curr_sample_weight = X, y, sample_weight
    
        X_transformed = random_projection.fit_transform(X_sample)
        model = DecisionTreeRegressor(
            criterion=self.criterion, splitter=self.splitter,
            max_depth=self.max_depth, min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf, min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features, random_state=local_rng,
            max_leaf_nodes=self.max_leaf_nodes, min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha
        )
        model.fit(X_transformed, y_sample, sample_weight=curr_sample_weight)
        return model, random_projection

    def fit(self, X, y, sample_weight=None):
        self.models_ = []
        self.projections_ = []
    
        if self.random_state is not None:
            rng = RandomState(self.random_state)
        else:
            rng = RandomState()
    
        random_states = [rng.randint(0, 1 << 30) for _ in range(self.n_estimators)]
    
        # _fit_single_treeを並列で実行
        results = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_single_tree)(X, y, sample_weight, rs) for rs in random_states)
    
        # 結果をmodels_とprojections_に格納
        self.models_, self.projections_ = zip(*results)
    
        return self
    
    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        
        for model, proj in zip(self.models_, self.projections_):
            X_transformed = proj.transform(X)  # 射影を適用
            predictions += model.predict(X_transformed)
        
        return predictions / self.n_estimators


# Normalize the sample_weight
def normalize_weights(weights):
    return weights / np.sum(weights)

class GRPForest(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=10, n_components='auto', eps=0.1, 
                 criterion='squared_error', splitter='best', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features=None, random_state=None, max_leaf_nodes=None,
                 min_impurity_decrease=0.0, ccp_alpha=0.0, n_jobs=1):
        self.n_estimators = n_estimators
        self.n_components = n_components
        self.eps = eps
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.n_jobs = n_jobs
        self.models_ = []
        self.random_projection_ = None

    def _fit_single_tree(self, seed, X, y, sample_weight):
        local_rng = RandomState(seed)
        if sample_weight is not None:
            indices = local_rng.choice(range(len(X)), size=len(X), p=sample_weight)
            new_sample_weight = np.zeros(len(X))
            for idx in indices:
                new_sample_weight[idx] += sample_weight[idx]
            X_bootstrap = X[indices, :]
            y_bootstrap = y[indices]
        else:
            X_bootstrap, y_bootstrap = resample(X, y, random_state=local_rng)
            new_sample_weight = None

        X_transformed = self.random_projection_.transform(X_bootstrap)
        model = DecisionTreeRegressor(
            criterion=self.criterion, splitter=self.splitter,
            max_depth=self.max_depth, min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf, min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features, random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes, min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha
        )
        model.fit(X_transformed, y_bootstrap, sample_weight=new_sample_weight)
        return model

    def fit(self, X, y, sample_weight=None):
        self.models_ = []
        rng = RandomState(self.random_state)
        
        target_n_components = min(X.shape[1], int(X.shape[1] * (1 - self.eps)))
        self.random_projection_ = GaussianRandomProjection(n_components=target_n_components, eps=self.eps, 
                                                           random_state=self.random_state)
        
        seeds = [rng.randint(0, 1 << 30) for _ in range(self.n_estimators)]
        
        self.models_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_tree)(seed, X, y, sample_weight) for seed in seeds
        )
        return self
    
    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        
        for model in self.models_:
            X_transformed = self.random_projection_.transform(X)
            predictions += model.predict(X_transformed)
        
        return predictions / self.n_estimators


class EnsembleDecisionTreeRegressorWithBootstrapSampleWeight(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=10, n_components='auto', eps=0.1, 
                 criterion='squared_error', splitter='best', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features=None, random_state=None, max_leaf_nodes=None,
                 min_impurity_decrease=0.0, ccp_alpha=0.0):
        self.n_estimators = n_estimators
        self.n_components = n_components
        self.eps = eps
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.models_ = []
        self.random_projection_ = None
        
    def fit(self, X, y, sample_weight=None):
        self.models_ = []
        np.random.seed(self.random_state)
        
        # Initialize GaussianRandomProjection
        target_n_components = min(X.shape[1], int(X.shape[1] * (1 - self.eps)))
        self.random_projection_ = GaussianRandomProjection(n_components=target_n_components, eps=self.eps, 
                                                           random_state=self.random_state)
        
        for i in range(self.n_estimators):
            # Bootstrap resampling with sample_weight
            if sample_weight is not None:
                indices = np.random.choice(range(len(X)), size=len(X), p=sample_weight)
                
                # Increase sample_weight for resampled instances
                new_sample_weight = np.zeros(len(X))
                for idx in indices:
                    new_sample_weight[idx] += sample_weight[idx]
                    
                X_bootstrap = X[indices, :]
                y_bootstrap = y[indices]
            else:
                X_bootstrap, y_bootstrap = resample(X, y, random_state=i + self.random_state)
                new_sample_weight = None
            
            # Perform Gaussian Random Projection
            X_transformed = self.random_projection_.fit_transform(X_bootstrap)
            
            # Train a Decision Tree Regressor on the transformed data
            model = DecisionTreeRegressor(
                criterion=self.criterion, splitter=self.splitter,
                max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf, min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features, random_state=self.random_state,
                max_leaf_nodes=self.max_leaf_nodes, min_impurity_decrease=self.min_impurity_decrease,
                ccp_alpha=self.ccp_alpha
            )
            model.fit(X_transformed, y_bootstrap, sample_weight=new_sample_weight)
            
            self.models_.append(model)
        return self
    
    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        
        for model in self.models_:
            X_transformed = self.random_projection_.transform(X)
            predictions += model.predict(X_transformed)
        
        return predictions / self.n_estimators

# Create sample data and train model
ensemble_model_with_bootstrap_weight = EnsembleDecisionTreeRegressorWithBootstrapSampleWeight(
    n_estimators=10, 
    n_components='auto', 
    eps=0.1, 
    max_depth=5,
    random_state=42
)

# Normalize sample weights
normalized_weights = normalize_weights(np.ones(y_train.shape[0]))

# Fit the model
ensemble_model_with_bootstrap_weight.fit(X_train, y_train, sample_weight=normalized_weights)

# Show first 5 predictions
ensemble_model_with_bootstrap_weight.predict(X_test)[:5]


class EnsembleDecisionTreeRegressorWithRandomProjectionAndEpsilon(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=10, n_components='auto', eps=0.1, 
                 criterion='squared_error', splitter='best', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features=None, random_state=None, max_leaf_nodes=None,
                 min_impurity_decrease=0.0, ccp_alpha=0.0):
        self.n_estimators = n_estimators
        self.n_components = n_components
        self.eps = eps
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.models_ = []
        self.random_projection_ = None
        
    def fit(self, X, y):
        self.models_ = []
        np.random.seed(self.random_state)
        
        # Initialize GaussianRandomProjection
        target_n_components = min(X.shape[1], int(X.shape[1] * (1 - self.eps)))
        self.random_projection_ = GaussianRandomProjection(n_components=target_n_components, eps=self.eps, 
                                                           random_state=self.random_state)
        
        for _ in range(self.n_estimators):
            # Perform Gaussian Random Projection
            X_transformed = self.random_projection_.fit_transform(X)
            
            # Train a Decision Tree Regressor on the transformed data
            model = DecisionTreeRegressor(
                criterion=self.criterion, splitter=self.splitter,
                max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf, min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features, random_state=self.random_state,
                max_leaf_nodes=self.max_leaf_nodes, min_impurity_decrease=self.min_impurity_decrease,
                ccp_alpha=self.ccp_alpha
            )
            model.fit(X_transformed, y)
            
            self.models_.append(model)
        return self
    
    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        
        for model in self.models_:
            X_transformed = self.random_projection_.transform(X)
            predictions += model.predict(X_transformed)
        
        return predictions / self.n_estimators

class EnsembleDecisionTreeRegressorWithRandomProjection(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=10, n_components=10, random_state=None):
        self.n_estimators = n_estimators
        self.n_components = n_components
        self.random_state = random_state
        self.models_ = []
        
    def _gaussian_random_projection(self, X):
        random_matrix = np.random.normal(size=(X.shape[1], self.n_components))
        return X.dot(random_matrix)
        
    def fit(self, X, y):
        self.models_ = []
        np.random.seed(self.random_state)
        
        for _ in range(self.n_estimators):
            # Perform Gaussian Random Projection
            X_transformed = self._gaussian_random_projection(X)
            
            # Train a Decision Tree Regressor on the transformed data
            model = DecisionTreeRegressor(random_state=self.random_state)
            model.fit(X_transformed, y)
            
            self.models_.append(model)
        return self
    
    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        
        for model in self.models_:
            X_transformed = self._gaussian_random_projection(X)
            predictions += model.predict(X_transformed)
        
        return predictions / self.n_estimators

class EnsembleDecisionTreeRegressorWithRandomProjectionAndEpsilon(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=10, n_components='auto', eps=0.1, random_state=None):
        self.n_estimators = n_estimators
        self.n_components = n_components
        self.eps = eps
        self.random_state = random_state
        self.models_ = []
        self.random_projection_ = None
        
    def fit(self, X, y):
        self.models_ = []
        np.random.seed(self.random_state)
        
        # Initialize GaussianRandomProjection
        target_n_components = min(X.shape[1], int(X.shape[1] * (1 - self.eps)))
        self.random_projection_ = GaussianRandomProjection(n_components=target_n_components, eps=self.eps, 
                                                           random_state=self.random_state)
        
        for _ in range(self.n_estimators):
            # Perform Gaussian Random Projection
            X_transformed = self.random_projection_.fit_transform(X)
            
            # Train a Decision Tree Regressor on the transformed data
            model = DecisionTreeRegressor(random_state=self.random_state)
            model.fit(X_transformed, y)
            
            self.models_.append(model)
        return self
    
    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        
        for model in self.models_:
            X_transformed = self.random_projection_.transform(X)
            predictions += model.predict(X_transformed)
        
        return predictions / self.n_estimators
