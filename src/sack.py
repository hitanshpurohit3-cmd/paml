import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

class SACKClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.models = [
            RandomForestClassifier(n_estimators=100),
            LogisticRegression(max_iter=1000)
        ]

    def fit(self, X, y):
        self.trained_models = []

        for model in self.models:
            model.fit(X, y)
            self.trained_models.append(model)

        return self

    def predict(self, X):
        preds = []

        for model in self.trained_models:
            preds.append(model.predict(X))

        preds = np.array(preds)

        # Majority voting
        final_preds = np.round(np.mean(preds, axis=0)).astype(int)
        return final_preds

    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)