import numpy as np

class DecisionTreeRegressorScratch:

    def __init__(self, max_depth=10, min_samples_split=5, n_split_candidates=25):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_split_candidates = n_split_candidates
        self.tree = None

    def mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def best_split_feature(self, X_col, y):
        thresholds = np.linspace(X_col.min(), X_col.max(), self.n_split_candidates)

        best_threshold = None
        best_mse = float("inf")

        for threshold in thresholds:
            left_mask = X_col < threshold
            right_mask = ~left_mask

            if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                continue

            left_y = y[left_mask]
            right_y = y[right_mask]

            mse_left = self.mse(left_y)
            mse_right = self.mse(right_y)

            total_mse = (len(left_y)*mse_left + len(right_y)*mse_right) / len(y)

            if total_mse < best_mse:
                best_mse = total_mse
                best_threshold = threshold

        return best_threshold, best_mse

    def best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_mse = float("inf")

        n_features = X.shape[1]

        for feature in range(n_features):
            threshold, mse_val = self.best_split_feature(X[:, feature], y)

            if threshold is not None and mse_val < best_mse:
                best_mse = mse_val
                best_threshold = threshold
                best_feature = feature

        return best_feature, best_threshold, best_mse

    def build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return np.mean(y)

        feature, threshold, mse_val = self.best_split(X, y)

        if feature is None:
            return np.mean(y)

        left_mask = X[:, feature] < threshold
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return np.mean(y)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": self.build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self.build_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def fit(self, X, y):
        self.tree = self.build_tree(X, y, depth=0)

    def predict_one(self, x, node):
        if not isinstance(node, dict):
            return node

        if x[node["feature"]] < node["threshold"]:
            return self.predict_one(x, node["left"])
        else:
            return self.predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])