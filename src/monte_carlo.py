import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class MonteCarlo:
    def __init__(self, n_splits=30, test_size=0.2):
        self.n_splits = n_splits
        self.test_size = test_size

    def run(self, model, X, y):
        scores = []

        for i in range(self.n_splits):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                stratify=y,
                random_state=i
            )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            scores.append(acc)

            print(f"Run {i+1}: Accuracy = {acc:.4f}")

        print("\nFinal Results:")
        print(f"Mean Accuracy: {np.mean(scores):.4f}")
        print(f"Std Dev: {np.std(scores):.4f}")

        return scores