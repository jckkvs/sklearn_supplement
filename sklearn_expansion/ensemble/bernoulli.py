from sklearn.utils import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class BRFTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 max_depth=None, 
                 max_features='sqrt', 
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 ccp_alpha=0.0,
                 bootstrap=True,
                 random_state=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.bootstrap = bootstrap
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        if self.max_features is not None:
            n_features = X.shape[1]
            if self.max_features == "sqrt":
                max_features_ = int(np.sqrt(n_features))
            elif isinstance(self.max_features, int):
                max_features_ = self.max_features
            elif isinstance(self.max_features, float):
                max_features_ = int(self.max_features * n_features)
            else:
                raise ValueError("Invalid value for max_features.")
            self.selected_features_ = np.random.choice(n_features, max_features_, replace=False)
        else:
            self.selected_features_ = np.arange(X.shape[1])      
        self.tree_ = self._fit_tree(X, y, sample_weight, depth=0, parent_cost=0)
        return self
    def _fit_tree(self, X, y, sample_weight, depth, parent_cost):
        total_weight = np.sum(np.ones(len(y)))  # サンプル重みの合計（ここでは1と仮定）
        min_weight = self.min_weight_fraction_leaf * total_weight  # 葉ノードに必要な最小のサンプル重み
        X_sub = X[:, self.selected_features_]
        # ノードに割り当てられたサンプルの重み合計が min_weight 以上かチェック
        if np.sum(np.ones(len(y))) < min_weight:
            return {"type": "leaf", "class": self._most_common_class(y)}

        if self.max_depth is not None and depth >= self.max_depth:
            return {"type": "leaf", "class": self._most_common_class(y)}

        if len(np.unique(y)) == 1:
            return {"type": "leaf", "class": y[0]}

        if len(y) < self.min_samples_split:
            return {"type": "leaf", "class": self._most_common_class(y)}

        if self.max_leaf_nodes is not None and self.current_leaf_nodes >= self.max_leaf_nodes:
            return {"type": "leaf", "class": self._most_common_class(y, sample_weight)}

        best_gain = 0.0
        best_feature = None
        best_threshold = None
        best_left_mask = None
        best_right_mask = None
        best_left_cost = 0.0
        best_right_cost = 0.0

        n_samples, n_features = X_sub.shape
        features = np.random.choice(n_features, n_features, replace=False)

        for feature in features:
            values = X_sub[:, feature]
            levels = np.unique(values)
            thresholds = (levels[:-1] + levels[1:]) / 2.0

            for threshold in thresholds:
                left_mask = values < threshold
                right_mask = values >= threshold

                left_y = y[left_mask]
                right_y = y[right_mask]
                left_sample_weight = sample_weight[left_mask]
                right_sample_weight = sample_weight[right_mask]
                gain = self._information_gain(y, left_y, right_y, sample_weight, left_sample_weight, right_sample_weight)
                left_cost = self._cost(left_y, gain)
                right_cost = self._cost(right_y, gain)

                total_cost = left_cost + right_cost + self.ccp_alpha

                if total_cost < parent_cost:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    best_left_mask = left_mask
                    best_right_mask = right_mask
                    best_left_cost = left_cost
                    best_right_cost = right_cost

        if best_gain == 0.0:
            return {"type": "leaf", "class": self._most_common_class(y)}

        left_tree = self._fit_tree(X_sub[best_left_mask], y[best_left_mask], depth + 1, best_left_cost)
        right_tree = self._fit_tree(X_sub[best_right_mask], y[best_right_mask], depth + 1, best_right_cost)

        return {
            "type": "node",
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left_tree,
            "right": right_tree,
        }

    def _most_common_class(self, y, sample_weight):
        unique_classes, class_counts = np.unique(y, return_counts=True)
        weighted_counts = class_counts * sample_weight
        return unique_classes[np.argmax(weighted_counts)]

def _information_gain(self, parent_y, left_child_y, right_child_y, parent_sample_weight, left_sample_weight, right_sample_weight):
    def weighted_entropy(y, sample_weight):
        hist = np.bincount(y, weights=sample_weight)
        ps = hist / np.sum(hist)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    parent_entropy = weighted_entropy(parent_y, parent_sample_weight)
    left_entropy = weighted_entropy(left_child_y, left_sample_weight)
    right_entropy = weighted_entropy(right_child_y, right_sample_weight)

    n = np.sum(parent_sample_weight)
    n_l = np.sum(left_sample_weight)
    n_r = np.sum(right_sample_weight)

    child_entropy = (n_l / n) * left_entropy + (n_r / n) * right_entropy
    return parent_entropy - child_entropy

    def _cost(self, y, gain):
        return -gain  # ここは独自のコスト関数に置き換えることが可能です

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        predictions = [self._predict_sample(sample, self.tree_) for sample in X]
        return np.array(predictions)

    def _predict_sample(self, sample, tree):
        if tree["type"] == "leaf":
            return tree["class"]

        if sample[tree["feature"]] < tree["threshold"]:
            return self._predict_sample(sample, tree["left"])
        else:
            return self._predict_sample(sample, tree["right"])

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from joblib import Parallel, delayed
import numpy as np
import random

class BRFRandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 n_estimators=100, 
                 max_depth=None, 
                 max_features='sqrt', 
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 ccp_alpha=0.0,
                 bootstrap=True,
                 n_jobs=None,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        # Input validation and setup
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        
        self.trees_ = []

        for i in range(self.n_estimators):
            if self.bootstrap:
                # Sample with replacement
                indices = np.random.choice(len(X), len(X), replace=True)
                X_sample, y_sample = X[indices], y[indices]
            else:
                X_sample, y_sample = X, y

            tree = BRFTreeClassifier(
                max_depth=self.max_depth, 
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                ccp_alpha=self.ccp_alpha,
            )
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        def predict_single_tree(tree):
            return tree.predict(X)
        
        predictions = Parallel(n_jobs=self.n_jobs)(delayed(predict_single_tree)(tree) for tree in self.trees_)
        
        # Majority vote
        predictions = np.array(predictions).T
        majority_votes = [np.bincount(row).argmax() for row in predictions]

        return np.array(majority_votes)

