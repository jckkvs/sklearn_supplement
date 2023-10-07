from sklearn.base import BaseEstimator, RegressorMixin

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

# サンプルデータとモデルの訓練
X, y = make_regression(n_samples=200, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ensemble_model = EnsembleDecisionTreeRegressorWithRandomProjection(n_estimators=10, n_components=10, random_state=42)
ensemble_model.fit(X_train, y_train)

# 予測結果の先頭5件を表示
ensemble_model.predict(X_test)[:5]
