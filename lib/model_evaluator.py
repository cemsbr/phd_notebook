"""Evaluate models."""
import numpy as np
import pandas as pd
from sklearn import metrics


class ModelEvaluator:
    """Evaluate models."""

    METRICS = (
        # Mean Absolute Error
        ('MAE', metrics.mean_absolute_error),
        # Mean Percentage Error
        ('MPE', lambda y, pred: (np.abs(y - pred) / y).mean()),
        # Root Mean Squared Error
        ('RMSE', lambda y, pred: metrics.mean_squared_error(y, pred)**0.5),
    )

    def __init__(self, features_csv):
        """Only csv file to be load DataFrame in worker."""
        self._features_csv = features_csv
        self._df_full = None
        self._apps = None
        #: DataFrame with only the model's features
        self._df = None
        self._model = None

    def evaluate(self, args):
        """Evaluate models."""
        creator, cpu, cpus = args
        self._init_worker()
        evals = []
        for model in creator.get_models(cpu, cpus):
            df_model = self._set_model(model)
            for app in self._apps:
                self._set_app(app, df_model)
                evals.append([app] + self._evaluate() + [model.number])
        return evals

    def _set_model(self, model):
        self._model = model
        cols = model.features + [model.y, 'set', 'application']
        return self._df_full[cols].replace([np.inf, -np.inf], np.nan).dropna()

    def _set_app(self, app, df_model):
        query = 'application == "{}"'.format(app)
        self._df = df_model.query(query)

    @classmethod
    def get_csv_header(cls):
        """Useful for CSV header."""
        return ['application'] + [title for title, _ in cls.METRICS] + \
               ['model']

    def _init_worker(self):
        """Initialize worker data."""
        self._df_full = pd.read_csv(self._features_csv)
        self._apps = sorted(self._df_full['application'].unique())

    def _evaluate(self):
        self._train()
        y, pred = self._predict()
        if self._model.is_log:
            y = 2**y
            pred = 2**pred
        return [fn(y, pred) for _, fn in self.METRICS]

    def _train(self):
        profiling = self._df.query('set == "profiling"')
        x = profiling[self._model.features]
        y = profiling[self._model.y]
        self._model.linear_model.fit(x, y)

    def _predict(self):
        target = self._df.query('set == "target"')
        x = target[self._model.features]
        y = target[self._model.y]
        predictions = self._model.linear_model.predict(x)
        return y, predictions
