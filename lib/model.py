"""The one and only module for the sake of documentation."""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


class Model:
    """Base class to use scikit-learn models."""

    def __init__(self, model, ycol='ms', degree=None, use_log=False):
        """Base class to use scikit-learn models.

        :param model: A scikit-learn model
        :param int degree: For polynomial expansion.
            Use fit_intercept=False in model constructor.
        :param bool use_log: Whether to use log scale.
            Note: predictions won't be in log scale.
        """
        self._ycol = ycol
        if degree is None:
            self.model = model
        else:
            self.model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)), (
                    'linear', model)
            ])
        self.use_log = use_log

    def fit(self, df):
        """Train the model."""
        features = df.drop(self._ycol, axis=1)
        y = df[self._ycol]
        if self.use_log:
            self.model.fit(np.log2(features), np.log2(y))
        else:
            self.model.fit(features, y)

    def predict(self, df):
        """After training, make predictions."""
        if isinstance(df, pd.DataFrame) and self._ycol in df.columns:
            features = df.drop(self._ycol, axis=1)
        else:
            features = df

        if self.use_log:
            log_predictions = self.model.predict(np.log2(features))
            predictions = np.power(2, log_predictions)
        else:
            predictions = self.model.predict(features)
        return predictions

    def score(self, df):
        """Calculate mean absolute percentage error.

        :param DataFrame df: Features and actual results
        """
        predictions = self.predict(df)
        real = df[self._ycol]
        errors = np.abs(predictions - real) / real
        scores = []
        scores.append(np.mean(errors))  # mean of all errors
        # grouped by worker amount and input size
        for workers, size in df[['workers', 'input']].drop_duplicates().values:
            select = (df.workers == workers) & (df.input == size)
            error = np.mean(errors[select])
            scores.append(((workers, size), error))

        return scores
