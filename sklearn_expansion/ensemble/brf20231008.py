# Displaying the entire BRFDecisionTreeRegressor class with all the methods and the build_tree function

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
