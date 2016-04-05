"""The one and only module for the sake of documentation."""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


class Model:
    """Base class to use scikit-learn models."""

    def __init__(self,
                 model,
                 features=None,
                 ycol='ms',
                 degree=None,
                 use_log=False):
        """Base class to use scikit-learn models.

        :param model: A scikit-learn model
        :param int degree: For polynomial expansion.
            Use fit_intercept=False in model constructor.
        :param bool use_log: Whether to use log scale.
            Note: predictions won't be in log scale.
        """
        self._features = features
        self._ycol = ycol
        if degree is None:
            self.model = model
        else:
            self.model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)), ('linear', model)
            ])
        self.use_log = use_log

    def fit(self, df):
        """Train the model."""
        features = self._get_features(df)
        y = df[self._ycol]
        if self.use_log:
            self.model.fit(np.log2(features), np.log2(y))
        else:
            self.model.fit(features, y)

    def predict(self, df):
        """After training, make predictions."""
        if isinstance(df, pd.DataFrame):
            features = self._get_features(df)
        else:
            features = df

        if self.use_log:
            log_predictions = self.model.predict(np.log2(features))
            predictions = 2**log_predictions
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
        for key, select in _get_score_keys(df):
            error = np.mean(errors[select])
            scores.append((key, error))

        return scores

    def fit_score(self, df):
        self.fit(df)
        return self.score(df)

    def set_intercept(self, df_match):
        current = self.predict(df_match)[0]
        target = df_match[self._ycol][0]
        linreg = self.get_linreg()
        if self.use_log:
            linreg.intercept_ += np.log2(target / current)
        else:
            linreg.intercept_ += target - current

    def get_linreg(self):
        if isinstance(self.model, Pipeline):
            linreg = self.model.named_steps['linear']
        else:
            linreg = self.model
        return linreg

    def _get_features(self, df):
        if self._features is None:
            self._features = [c for c in df.columns if c != self._ycol]
        return df[self._features]


def _get_score_keys(df):
    if 'workers' in df.columns and 'input' in df.columns:
        for workers, size in df[['workers', 'input']].drop_duplicates().values:
            key = (workers, size)
            select = (df.workers == workers) & (df.input == size)
            yield (key, select)
    elif 'workers' not in df.columns:
        for size in df['input'].unique():
            key = size
            select = df.input == size
            yield (key, select)
    else:
        for workers in df['workers'].unique():
            key = workers
            select = df.workers == workers
            yield (key, select)
