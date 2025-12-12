import numpy as np
from src.models.decision_tree import DecisionTreeRegressorScratch

class RandomForestRegressorScratch:
    np.random.seed(42)
    def __init__(self, n_trees=20, max_depth=10, min_samples_split=5, max_features="sqrt", random_state=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        np.random.seed(random_state)  # FIXED SEED
        self.trees = []

    def bootstrap(self, X, y):
        n = len(X)
        idx = np.random.choice(n, n, replace=True)
        return X[idx], y[idx]

    def choose_features(self, n_features):
        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            k = max(1, int(np.log2(n_features)))
        else:
            k = n_features

        # choose WITHOUT replacement (IMPORTANT)
        return np.random.choice(n_features, k, replace=False)

    def fit(self, X, y):
        self.trees = []
        n_features = X.shape[1]

        for _ in range(self.n_trees):
            X_s, y_s = self.bootstrap(X, y)

            feat_idx = self.choose_features(n_features)

            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )

            tree.fit(X_s[:, feat_idx], y_s)

            self.trees.append((tree, feat_idx))

    def predict(self, X):
        preds = []

        for tree, feat_idx in self.trees:
            preds.append(tree.predict(X[:, feat_idx]))

        return np.mean(preds, axis=0)
