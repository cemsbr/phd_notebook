import numpy as np


class Model:
    """Base class to use scikit-learn models."""

    def __init__(self, model, use_log=False):
        """Base class to use scikit-learn models.

        :param model: A scikit-learn model
        :param bool use_log: Whether to use log scale.
            Note: predictions won't be in log scale.
        """
        self.model = model
        self.use_log = use_log

    def train(self, X, y):
        """Train the model (as in scikit-learn docs)."""
        if self.use_log:
            self.model.fit(np.log(X), np.log(y))
        else:
            self.model.fit(X, y)

    def predict(self, X):
        """After training, make predictions (as in scikit-learn docs)."""
        if self.use_log:
            log_predictions = self.model.predict(np.log(X))
            predictions = np.exp(log_predictions)
        else:
            predictions = self.model.predict(X)
        return predictions

    def score(self, X, y):
        """Calculate mean absolute percentage error.

        :param X: Test samples
        :param y: True values for X
        """
        predictions = self.predict(X)
        return np.mean(np.abs(predictions - y) / y)
