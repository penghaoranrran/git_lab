import numpy as np
import pandas as pd
from collections import Counter
import math

class Node:
    def __init__(self, feature=None, threshold=None, value=None, is_leaf=False):
        self.feature = feature        # Feature index or name to split on
        self.threshold = threshold    # Threshold for continuous features
        self.value = value            # Class label if leaf
        self.is_leaf = is_leaf        # Boolean
        self.children = {}            # Dictionary mapping feature values to child Nodes

class DecisionTreeClassifier:
    def __init__(self, criterion='id3', max_depth=None, min_samples_split=2, pruning=None):
        self.criterion = criterion  # 'id3', 'c45', 'cart'
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.pruning = pruning # 'pre', 'post', or None
        self.root = None
        self.feature_importances_ = {}
        
    def _entropy(self, y):
        counter = Counter(y)
        total = len(y)
        if total == 0: return 0
        ent = 0.0
        for count in counter.values():
            p = count / total
            ent -= p * math.log2(p)
        return ent
    
    def _gini(self, y):
        counter = Counter(y)
        total = len(y)
        if total == 0: return 0
        gini = 1.0
        for count in counter.values():
            p = count / total
            gini -= p ** 2
        return gini

    def _split_continuous(self, X_col, y):
        unique_values = sorted(list(set(X_col)))
        best_gain = -float('inf')
        best_threshold = None
        
        if len(unique_values) < 2:
            return None, -float('inf')
            
        split_points = [(unique_values[i] + unique_values[i+1])/2 for i in range(len(unique_values)-1)]
        
        base_score = self._entropy(y) if self.criterion != 'cart' else self._gini(y)
        
        for threshold in split_points:
            left_mask = X_col <= threshold
            right_mask = X_col > threshold
            
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            if len(y_left) == 0 or len(y_right) == 0:
                continue
                
            p_left = len(y_left) / len(y)
            p_right = len(y_right) / len(y)
            
            if self.criterion == 'cart':
                gain = base_score - (p_left * self._gini(y_left) + p_right * self._gini(y_right))
            else:
                gain = base_score - (p_left * self._entropy(y_left) + p_right * self._entropy(y_right))
                
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
                
        return best_threshold, best_gain

    def _choose_best_feature(self, X, y, features):
        best_feature = None
        best_gain = -float('inf')
        best_threshold = None
        
        base_score = self._entropy(y) if self.criterion != 'cart' else self._gini(y)
        
        for feature in features:
            X_col = X[feature]
            is_continuous = pd.api.types.is_float_dtype(X_col)
            
            if is_continuous:
                threshold, gain = self._split_continuous(X_col, y)
                # For continuous, we typically use Gain. 
                # C4.5 uses Gain Ratio for selecting the split attribute, but Gain for the cut point.
                # Here we simplify: if criterion is C4.5, we might check split info later if we want strict C4.5
                pass
            else:
                # Discrete
                values = X_col.unique()
                new_score = 0.0
                split_info = 0.0
                for v in values:
                    sub_y = y[X_col == v]
                    p = len(sub_y) / len(y)
                    if self.criterion == 'cart':
                        new_score += p * self._gini(sub_y)
                    else:
                        new_score += p * self._entropy(sub_y)
                        if p > 0:
                            split_info -= p * math.log2(p)
                
                gain = base_score - new_score
                threshold = None
                
                if self.criterion == 'c45':
                    if split_info == 0:
                        gain = 0
                    else:
                        gain = gain / split_info
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
                
        return best_feature, best_threshold, best_gain

    def fit(self, X, y, X_val=None, y_val=None):
        self.feature_importances_ = {f: 0.0 for f in X.columns}
        self.root = self._build_tree(X, y, depth=0)
        
        if self.pruning == 'post' and X_val is not None and y_val is not None:
             self._post_prune(self.root, X_val, y_val)

    def _build_tree(self, X, y, depth):
        # Stop conditions
        if len(set(y)) == 1:
            return Node(value=y.iloc[0], is_leaf=True)
        
        if X.shape[1] == 0 or (X.duplicated(keep=False).all() and len(X.drop_duplicates()) == 1):
             return Node(value=Counter(y).most_common(1)[0][0], is_leaf=True)

        if self.pruning == 'pre': # Check pre-pruning conditions only if enabled
            if self.max_depth is not None and depth >= self.max_depth:
                return Node(value=Counter(y).most_common(1)[0][0], is_leaf=True)
            if len(y) < self.min_samples_split:
                return Node(value=Counter(y).most_common(1)[0][0], is_leaf=True)
        # However, max_depth is often applied regardless of pruning strategy name as a safety.
        # Let's keep max_depth always active if set.
        if self.max_depth is not None and depth >= self.max_depth:
             return Node(value=Counter(y).most_common(1)[0][0], is_leaf=True)
        if self.min_samples_split is not None and len(y) < self.min_samples_split:
             return Node(value=Counter(y).most_common(1)[0][0], is_leaf=True)

        features = X.columns
        best_feature, best_threshold, best_gain = self._choose_best_feature(X, y, features)
        
        if best_feature is None or best_gain <= 0: # Stop if no gain
             return Node(value=Counter(y).most_common(1)[0][0], is_leaf=True)

        # Update feature importance
        # Importance = gain * N_t / N_total (simplified here, just accumulating gain)
        if best_feature in self.feature_importances_:
             self.feature_importances_[best_feature] += best_gain * len(y)

        node = Node(feature=best_feature, threshold=best_threshold)
        # Store most common label in case we need to turn this node into a leaf later (for pruning)
        node.value = Counter(y).most_common(1)[0][0] 
        
        if best_threshold is not None:
            left_mask = X[best_feature] <= best_threshold
            right_mask = X[best_feature] > best_threshold
            node.children['<='] = self._build_tree(X[left_mask], y[left_mask], depth + 1)
            node.children['>'] = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        else:
            values = X[best_feature].unique()
            for v in values:
                mask = X[best_feature] == v
                sub_X = X[mask].drop(columns=[best_feature])
                sub_y = y[mask]
                
                if len(sub_y) == 0:
                    node.children[v] = Node(value=Counter(y).most_common(1)[0][0], is_leaf=True)
                else:
                    node.children[v] = self._build_tree(sub_X, sub_y, depth + 1)
                    
        return node

    def _post_prune(self, node, X_val, y_val):
        if node.is_leaf:
            return

        # Prune children first
        for key, child in node.children.items():
            if not child.is_leaf:
                 # We need to filter X_val, y_val to match the child's branch
                 # This is tricky recursively efficiently.
                 # Actually, standard way is to evaluate accuracy on the *entire* validation set.
                 # But we need to know if making *this* node a leaf improves accuracy.
                 pass
        
        # Simple recursion:
        # First recurse down
        if node.threshold is not None:
             left_mask = X_val[node.feature] <= node.threshold
             right_mask = X_val[node.feature] > node.threshold
             if not node.children['<='].is_leaf:
                 self._post_prune(node.children['<='], X_val[left_mask], y_val[left_mask])
             if not node.children['>'].is_leaf:
                 self._post_prune(node.children['>'], X_val[right_mask], y_val[right_mask])
        else:
             # Discrete
             # If value not in children, we can't route data there.
             # This makes filtering hard.
             # Alternative: Just evaluate accuracy before and after pruning *this* node using the global predict.
             # But modifying the tree temporarily is expensive.
             # Correct bottom-up approach:
             # 1. Recurse to children.
             # 2. Check if children are now leaves (or if we are at bottom).
             # 3. Calculate error if we keep the subtree vs error if we turn this node into a leaf.
             pass
             
        # Because passing filtered data is complex for discrete with many values, 
        # let's try a simpler approach:
        # 1. Prune children
        if node.threshold is not None:
            child_keys = ['<=', '>']
        else:
            child_keys = list(node.children.keys())
            
        for key in child_keys:
             child = node.children[key]
             if not child.is_leaf:
                  # Filter data for child
                  if node.threshold is not None:
                      mask = (X_val[node.feature] <= node.threshold) if key == '<=' else (X_val[node.feature] > node.threshold)
                  else:
                      mask = (X_val[node.feature] == key)
                  
                  if len(y_val[mask]) > 0:
                      self._post_prune(child, X_val[mask], y_val[mask])

        # 2. Now check if we should prune this node
        # Calculate accuracy if we keep it
        y_pred_tree = self.predict(X_val) # Note: this uses the current tree (children might be pruned already)
        acc_tree = np.mean(y_pred_tree == y_val)
        
        # Calculate accuracy if we prune it (make it a leaf)
        # Temporarily make it a leaf
        original_is_leaf = node.is_leaf
        node.is_leaf = True
        
        y_pred_leaf = self.predict(X_val)
        acc_leaf = np.mean(y_pred_leaf == y_val)
        
        if acc_leaf >= acc_tree:
             # Keep it as leaf
             # Discard children
             node.children = {}
        else:
             # Revert
             node.is_leaf = original_is_leaf

    def predict_one(self, row, node=None):
        if node is None:
            node = self.root
        
        if node.is_leaf:
            return node.value
            
        if node.threshold is not None:
            val = row[node.feature]
            if val <= node.threshold:
                return self.predict_one(row, node.children['<='])
            else:
                return self.predict_one(row, node.children['>'])
        else:
            val = row[node.feature]
            if val in node.children:
                return self.predict_one(row, node.children[val])
            else:
                # Missing branch, return the majority class of the current node
                # We stored it in node.value during build
                return node.value 

    def predict(self, X):
        return X.apply(lambda row: self.predict_one(row), axis=1)

    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.root
            
        if node.is_leaf:
            print(f"{indent}Label: {node.value}")
            return
            
        if node.threshold is not None:
            print(f"{indent}{node.feature} <= {node.threshold:.3f} ?")
            self.print_tree(node.children['<='], indent + "  ")
            self.print_tree(node.children['>'], indent + "  ")
        else:
            print(f"{indent}{node.feature} ?")
            print("B31 experimental version")
            print("C4 experimental version")
            for val, child in node.children.items():
                self.print_tree(child, indent + "    ")
