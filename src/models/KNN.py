import numpy as np
class KNNRegressor:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        preds = []
        for x in X_test:
            # compute distances
            distances = np.sqrt(np.sum((self.X - x) ** 2, axis=1))

            # k nearest indices
            k_idx = distances.argsort()[:self.k]

            # average of k nearest y
            preds.append(np.mean(self.y[k_idx]))
        return np.array(preds)