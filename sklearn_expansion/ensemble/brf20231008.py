# Displaying the entire BRFDecisionTreeRegressor class with all the methods and the build_tree function

コード詳細を隠す
python
Copy code
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Define the BRFDecisionTreeRegressor class with the missing _predict_single method
class BRFDecisionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, min_samples_leaf=5, p1=0.5, p2=0.5, criterion='mse', kn=5):
        self.min_samples_leaf = min_samples_leaf
        self.p1 = p1
        self.p2 = p2
        self.criterion = criterion
        self.kn = kn  # Added kn parameter for stopping condition
        self.tree = None

    def fit(self, X, y):
        # Randomly partition the data into Structure and Estimation parts
        n_samples = len(y)
        idx = np.random.permutation(n_samples)
        structure_idx = idx[:n_samples // 2]
        estimation_idx = idx[n_samples // 2:]
        X_structure, y_structure = X[structure_idx], y[structure_idx]
        X_estimation, y_estimation = X[estimation_idx], y[estimation_idx]

        # Build the tree
        self.tree = self.build_tree(X_structure, y_structure, X_estimation, y_estimation, 
                                    depth=0, min_samples_leaf=self.min_samples_leaf, 
                                    p1=self.p1, p2=self.p2, criterion=self.criterion)

    def build_tree(self, X_structure, y_structure, X_estimation, y_estimation, depth, min_samples_leaf, p1, p2, criterion):
        # Stopping condition based on Estimation part
        if len(y_estimation) <= self.kn:
            return {'type': 'leaf', 'value': np.mean(y_estimation) if y_estimation.size > 0 else 0.0}
        
        n_samples, n_features = X_structure.shape

        # Stopping condition based on min_samples_leaf and unique labels
        if n_samples <= min_samples_leaf or np.all(y_structure == y_structure[0]):
            return {'type': 'leaf', 'value': np.mean(y_estimation) if y_estimation.size > 0 else 0.0}

        best_split = {'gain': np.inf}

        # Feature selection based on p1
        if np.random.rand() < p1:
            candidate_features = np.random.choice(n_features, 1)
        else:
            candidate_features = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
        
        # Loop through candidate features to find the best split
        for feature_index in candidate_features:
            feature_values = X_structure[:, feature_index]
            unique_values = np.unique(feature_values)
            if unique_values.size < 2:
                continue
            
            # Splitting point selection based on p2
            if np.random.rand() < p2:
                split_values = [np.random.choice(unique_values)]
            else:
                split_values = (unique_values[:-1] + unique_values[1:]) / 2

            # Calculate gain for each split value
            for sv in split_values:
                left_mask = feature_values < sv
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                if criterion == 'mse':
                    left_impurity = np.mean((y_structure[left_mask] - np.mean(y_structure[left_mask]))**2)
                    right_impurity = np.mean((y_structure[right_mask] - np.mean(y_structure[right_mask]))**2)
                elif criterion == 'mae':
                    left_impurity = np.mean(np.abs(y_structure[left_mask] - np.mean(y_structure[left_mask])))
                    right_impurity = np.mean(np.abs(y_structure[right_mask] - np.mean(y_structure[right_mask])))

                gain = (left_impurity * np.sum(left_mask) + right_impurity * np.sum(right_mask)) / n_samples

                if gain < best_split['gain']:
                    best_split = {'gain': gain, 'feature_index': feature_index, 'split_value': sv,
                                  'left_mask': left_mask, 'right_mask': right_mask}

        if best_split['gain'] == np.inf:
            return {'type': 'leaf', 'value': np.mean(y_estimation)}
        
        # Recursively build left and right subtrees
        left_tree = self.build_tree(X_structure[best_split['left_mask']], y_structure[best_split['left_mask']],
                                    X_estimation[best_split['left_mask']], y_estimation[best_split['left_mask']],
                                    depth + 1, min_samples_leaf, p1, p2, criterion)
        right_tree = self.build_tree(X_structure[best_split['right_mask']], y_structure[best_split['right_mask']],
                                     X_estimation[best_split['right_mask']], y_estimation[best_split['right_mask']],
                                     depth + 1, min_samples_leaf, p1, p2, criterion)

        return {'type': 'node', 'feature_index': best_split['feature_index'], 'split_value': best_split['split_value'],
                'left_tree': left_tree, 'right_tree': right_tree}

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self._predict_single(x, self.tree))
        return np.array(predictions)

    def _predict_single(self, x, tree):
        if tree['type'] == 'leaf':
            return tree['value']
        if x[tree['feature_index']] < tree['split_value']:
            return self._predict_single(x, tree['left_tree'])
        else:
            return self._predict_single(x, tree['right_tree'])



class BRFRandomForestRegressorWithRandomState(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, min_samples_leaf=5, p1=0.25, p2=0.25, criterion='mse', n_jobs=1, random_state=None):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.p1 = p1
        self.p2 = p2
        self.criterion = criterion
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.trees = []
        
    def _fit_single_tree(self, X, y, random_state):
        # Bootstrap sampling
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_sample, y_sample = X[indices], y[indices]

        # Create and fit a new BRF decision tree with unique random_state
        tree = BRFDecisionTreeRegressor(min_samples_leaf=self.min_samples_leaf,
                                        p1=self.p1, p2=self.p2, criterion=self.criterion)
        tree.random_state = random_state
        tree.fit(X_sample, y_sample)
        return tree
        
    def fit(self, X, y):
        # Initialize RandomState object based on self.random_state
        random_state = RandomState(self.random_state)
        
        # Fit trees in parallel, each with a unique random_state
        self.trees = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_single_tree)(X, y, random_state.randint(1000000)) for _ in range(self.n_estimators))
            
    def predict(self, X):
        # Average predictions from all trees
        predictions = np.mean(Parallel(n_jobs=self.n_jobs)(delayed(tree.predict)(X) for tree in self.trees), axis=0)
        return predictions

class BRFDecisionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, min_samples_leaf=5, p1=0.5, p2=0.5, criterion='mse'):
        self.min_samples_leaf = min_samples_leaf
        self.p1 = p1
        self.p2 = p2
        self.criterion = criterion
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self.build_tree_fixed(X, y, depth=0, min_samples_leaf=self.min_samples_leaf, 
                                          p1=self.p1, p2=self.p2, criterion=self.criterion)
        
    def build_tree_fixed(self, X, y, depth, min_samples_leaf, p1, p2, criterion):
        n_samples, n_features = X.shape

        if n_samples <= min_samples_leaf or np.all(y == y[0]):
            return {'type': 'leaf', 'value': np.mean(y) if y.size > 0 else 0.0}

        best_split = {'gain': np.inf}

        if np.random.rand() < p1:
            candidate_features = np.random.choice(n_features, 1)
        else:
            candidate_features = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
        
        for feature_index in candidate_features:
            feature_values = X[:, feature_index]
            unique_values = np.unique(feature_values)
            if unique_values.size < 2:
                continue
            
            # Implement p2 for random split point selection
            if np.random.rand() < p2:
                split_values = [np.random.choice(unique_values)]
            else:
                split_values = (unique_values[:-1] + unique_values[1:]) / 2

            for sv in split_values:
                left_mask = feature_values < sv
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                if criterion == 'mse':
                    left_impurity = np.mean((y[left_mask] - np.mean(y[left_mask]))**2)
                    right_impurity = np.mean((y[right_mask] - np.mean(y[right_mask]))**2)
                elif criterion == 'mae':
                    left_impurity = np.mean(np.abs(y[left_mask] - np.mean(y[left_mask])))
                    right_impurity = np.mean(np.abs(y[right_mask] - np.mean(y[right_mask])))

                gain = (left_impurity * np.sum(left_mask) + right_impurity * np.sum(right_mask)) / n_samples

                if gain < best_split['gain']:
                    best_split = {'gain': gain, 'feature_index': feature_index, 'split_value': sv,
                                  'left_mask': left_mask, 'right_mask': right_mask}

        if best_split['gain'] == np.inf:
            return {'type': 'leaf', 'value': np.mean(y)}

        left_tree = self.build_tree_fixed(X[best_split['left_mask']], y[best_split['left_mask']], depth+1,
                                         min_samples_leaf, p1, p2, criterion)
        right_tree = self.build_tree_fixed(X[best_split['right_mask']], y[best_split['right_mask']], depth+1,
                                          min_samples_leaf, p1, p2, criterion)

        return {'type': 'node', 'feature_index': best_split['feature_index'], 'split_value': best_split['split_value'],
                'left_tree': left_tree, 'right_tree': right_tree}

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self._predict_single(x, self.tree))
        return np.array(predictions)
    
    def _predict_single(self, x, tree):
        if tree['type'] == 'leaf':
            return tree['value']
        if x[tree['feature_index']] < tree['split_value']:
            return self._predict_single(x, tree['left_tree'])
        else:
            return self._predict_single(x, tree['right_tree'])

class BRFDecisionTreeRegressor:
    
    def __init__(self, min_samples_leaf=1, p1=0.5, p2=0.5, criterion='mse'):
        self.min_samples_leaf = min_samples_leaf
        self.p1 = p1
        self.p2 = p2
        self.criterion = criterion
        self.tree = None
    
    def fit(self, X, y):
        self.tree = build_tree(X, y, depth=0, min_samples_leaf=self.min_samples_leaf, 
                               p1=self.p1, p2=self.p2, criterion=self.criterion)
    
    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self._predict_single(x, self.tree))
        return np.array(predictions)
    
    def _predict_single(self, x, tree):
        if tree['type'] == 'leaf':
            return tree['value']
        if x[tree['feature_index']] < tree['split_value']:
            return self._predict_single(x, tree['left_tree'])
        else:
            return self._predict_single(x, tree['right_tree'])

# The build_tree function
def build_tree(X, y, depth, min_samples_leaf, p1, p2, criterion):
    n_samples, n_features = X.shape
    
    # Base case: if only one sample or all samples have the same target value
    if n_samples <= min_samples_leaf or np.all(y == y[0]):
        return {'type': 'leaf', 'value': np.mean(y)}
    
    # Placeholder for best split parameters
    best_split = {'gain': -np.inf}
    
    # Step 2: Select candidate features
    if np.random.rand() < p1:
        # Choose one candidate feature
        candidate_features = np.random.choice(n_features, 1)
    else:
        # Choose sqrt(D) candidate features
        candidate_features = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
    
    for feature_index in candidate_features:
        feature_values = X[:, feature_index]
        
        # Initialize split_value to None for this iteration
        split_value = None
        
        # Step 3: Select splitting point s
        if np.random.rand() < p2:
            # Randomly sample a point to split
            split_value = np.random.choice(feature_values)
        else:
            # Find the best splitting point by optimizing the criterion
            unique_values = np.unique(feature_values)
            split_values = (unique_values[:-1] + unique_values[1:]) / 2  # Midpoints
            
            for sv in split_values:
                left_mask = feature_values < sv
                right_mask = ~left_mask
                
                # Calculate impurity
                if criterion == 'mse':
                    left_impurity = mean_squared_error(y[left_mask], [np.mean(y[left_mask])] * np.sum(left_mask))
                    right_impurity = mean_squared_error(y[right_mask], [np.mean(y[right_mask])] * np.sum(right_mask))
                elif criterion == 'mae':
                    left_impurity = mean_absolute_error(y[left_mask], [np.mean(y[left_mask])] * np.sum(left_mask))
                    right_impurity = mean_absolute_error(y[right_mask], [np.mean(y[right_mask])] * np.sum(right_mask))
                
                # Calculate gain
                gain = left_impurity * np.sum(left_mask) + right_impurity * np.sum(right_mask)
                
                if gain < best_split['gain']:
                    best_split = {'gain': gain, 'feature_index': feature_index, 'split_value': sv,
                                  'left_mask': left_mask, 'right_mask': right_mask}
        
        # If a random split point was chosen, it should be utilized here
        if split_value is not None:
            best_split['split_value'] = split_value
            left_mask = feature_values < split_value
            right_mask = ~left_mask
            best_split['left_mask'] = left_mask
            best_split['right_mask'] = right_mask
    
    # Step 4: Create child nodes and proceed to the next level
    if best_split['gain'] == -np.inf:
        # No valid split was found
        return {'type': 'leaf', 'value': np.mean(y)}
    
    # Recursively build the tree
    left_tree = build_tree(X[best_split['left_mask']], y[best_split['left_mask']], depth+1,
                           min_samples_leaf, p1, p2, criterion)
    right_tree = build_tree(X[best_split['right_mask']], y[best_split['right_mask']], depth+1,
                            min_samples_leaf, p1, p2, criterion)
    
    return {'type': 'node', 'feature_index': best_split['feature_index'], 'split_value': best_split['split_value'],
            'left_tree': left_tree, 'right_tree': right_tree}
