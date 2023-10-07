from sklearn.base import BaseEstimator, RegressorMixin

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
