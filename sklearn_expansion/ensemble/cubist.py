class CubistRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, min_samples_leaf=2, max_depth=None, n_trees=10, random_state=None):
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.n_trees = n_trees  # Number of trees in the ensemble
        self.random_state = random_state
        self.trees = []  # List to store each tree

    def fit(self, X, y):
        self.trees = []  # Initialize the list of trees
        for _ in range(self.n_trees):
            tree = self.build_tree(X, y, depth=0)
            self.trees.append(tree)

    def build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        # Check the stopping conditions
        if n_samples <= self.min_samples_leaf or (self.max_depth and depth == self.max_depth):
            model = LinearRegression()
            model.fit(X, y)
            return {'type': 'leaf', 'model': model}
        
        # Find the best split
        best_split = self.find_best_split(X, y)
        if best_split is None:
            model = LinearRegression()
            model.fit(X, y)
            return {'type': 'leaf', 'model': model}
        
        # Recursively build the left and right parts of the tree
        left_tree = self.build_tree(best_split['left_X'], best_split['left_y'], depth + 1)
        right_tree = self.build_tree(best_split['right_X'], best_split['right_y'], depth + 1)
        
        return {
            'type': 'node',
            'feature_index': best_split['feature_index'],
            'threshold': best_split['threshold'],
            'left': left_tree,
            'right': right_tree
        }

    def find_best_split(self, X, y):
        # Initialize variables to store the best split found so far
        best_split = None
        best_score = float('inf')
        
        n_samples, n_features = X.shape
        
        for feature_index in range(n_features):
            # Extract values of the feature at the current index
            feature_values = X[:, feature_index]
            
            for threshold in feature_values:
                # Split data and labels based on the current threshold
                left_mask = feature_values < threshold
                right_mask = ~left_mask
                
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                # Skip this split if it would result in empty leaves
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                
                # Calculate the criterion for the current split
                left_score = mean_squared_error(left_y, [left_y.mean()] * len(left_y))
                right_score = mean_squared_error(right_y, [right_y.mean()] * len(right_y))
                current_score = (len(left_y) * left_score + len(right_y) * right_score) / n_samples
                
                # Update best split if current split is better
                if current_score < best_score:
                    best_score = current_score
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_X': X[left_mask],
                        'left_y': y[left_mask],
                        'right_X': X[right_mask],
                        'right_y': y[right_mask]
                    }
                    
        return best_split

    def predict(self, X):
        predictions = [self._predict_single_tree(tree, x) for x in X for tree in self.trees]
        return np.mean(predictions, axis=0)

    def _predict_single_tree(self, tree, x):
        if tree['type'] == 'leaf':
            return tree['model'].predict([x])[0]
        
        feature_index = tree['feature_index']
        threshold = tree['threshold']
        
        if x[feature_index] < threshold:
            return self._predict_single_tree(tree['left'], x)
        else:
            return self._predict_single_tree(tree['right'], x)
