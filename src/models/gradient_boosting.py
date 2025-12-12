import numpy as np
from src.models.decision_tree import DecisionTreeRegressorScratch

class GradientBoostingRegressorScratch:

    def __init__(
        self,
        n_estimators=80,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=2
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.initial_pred = None

    def fit(self, X, y):
        self.initial_pred = np.mean(y)
        y_pred = np.full(len(y), self.initial_pred)

        for _ in range(self.n_estimators):

            residuals = y - y_pred

            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_split_candidates=50
            )

            tree.fit(X, residuals)

            update = tree.predict(X)
            y_pred += self.learning_rate * update

            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.initial_pred)

        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred
