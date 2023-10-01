from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from scipy import stats
import numpy as np

# SPORFTree for classification and regression
class SPORFTree:
    def __init__(self, max_depth=None, random_state=None):
        self.max_depth = max_depth
        self.random_state = random_state
        self.tree = {}
        self.feature_importances_ = None

    def fit(self, X, y, task='classification', depth=0, node_id=0):
        n_samples, n_features = X.shape
        unique_values = np.unique(y)

        if len(unique_values) == 1 or (self.max_depth is not None and depth == self.max_depth):
            self.tree[node_id] = np.mean(y) if task == 'regression' else stats.mode(y)[0][0]
            return

        sparse_proj = np.random.randn(n_features)
        projected_X = X @ sparse_proj
        best_threshold = self._find_best_oblique_threshold(projected_X, y, task)
        
        if best_threshold is None:
            self.tree[node_id] = np.mean(y) if task == 'regression' else stats.mode(y)[0][0]
            return

        self.tree[node_id] = (sparse_proj, best_threshold)

        if self.feature_importances_ is None:
            self.feature_importances_ = np.zeros(n_features)
        self.feature_importances_ += np.abs(sparse_proj)

        left_mask = projected_X < best_threshold
        right_mask = ~left_mask
        self.fit(X[left_mask], y[left_mask], task, depth=depth+1, node_id=node_id*2+1)
        self.fit(X[right_mask], y[right_mask], task, depth=depth+1, node_id=node_id*2+2)

    def _find_best_oblique_threshold(self, projected_X, y, task):
        unique_values = np.unique(projected_X)
        best_metric = float('inf') if task == 'classification' else float('-inf')
        best_threshold = None

        for threshold in unique_values:
            left_mask = projected_X < threshold
            right_mask = ~left_mask

            if task == 'classification':
                metric_left = self._calculate_gini(y[left_mask])
                metric_right = self._calculate_gini(y[right_mask])
                metric = (metric_left * np.sum(left_mask) + metric_right * np.sum(right_mask)) / len(y)
            else:  # regression
                metric_left = np.mean((y[left_mask] - np.mean(y[left_mask])) ** 2)
                metric_right = np.mean((y[right_mask] - np.mean(y[right_mask])) ** 2)
                metric = (metric_left * np.sum(left_mask) + metric_right * np.sum(right_mask)) / len(y)

            if (task == 'classification' and metric < best_metric) or (task == 'regression' and metric > best_metric):
                best_metric = metric
                best_threshold = threshold
                
        return best_threshold

    def _calculate_gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / np.sum(counts)
        return 1 - np.sum(probabilities ** 2)

    def predict(self, X):
        return np.array([self._predict_sample(x, node_id=0) for x in X])

    def _predict_sample(self, x, node_id):
        node = self.tree.get(node_id)
        if node is None or not isinstance(node, tuple):
            return node
        sparse_proj, threshold = node
        projected_val = x @ sparse_proj
        if projected_val < threshold:
            return self._predict_sample(x, node_id=node_id*2+1)
        else:
            return self._predict_sample(x, node_id=node_id*2+2)

# SPORF for classification
class SPORFClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.feature_importances_ = np.zeros(X.shape[1])
        for _ in range(self.n_estimators):
            tree = SPORFTree(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X, y, task='classification')
            self.trees.append(tree)
            self.feature_importances_ += tree.feature_importances_
        self.feature_importances_ /= self.n_estimators
        return self

    def predict(self, X):
        predictions = [tree.predict(X) for tree in self.trees]
        return stats.mode(np.array(predictions), axis=0)[0][0]

# SPORF for regression
class SPORFRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=10, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        self.feature_importances_ = np.zeros(X.shape[1])
        for _ in range(self.n_estimators):
            tree = SPORFTree(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X, y, task='regression')
            self.trees.append(tree)
            self.feature_importances_ += tree.feature_importances_
        self.feature_importances_ /= self.n_estimators
        return self

    def predict(self, X):
        predictions = [tree.predict(X) for tree in self.trees]
        return np.mean(predictions, axis=0)

# Display the complete code
print("Code displayed without any omissions.")
