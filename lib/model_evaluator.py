"""Evaluate models."""
import numpy as np
import pandas as pd
from sklearn import metrics

from lib.csv_gen import CSVGen


class ModelEvaluator:
    """Evaluate models."""

    METRICS = (
        # Mean Absolute Percentage Error
        ('MAPE', lambda y, pred: (np.abs(y - pred) / y).mean()),
        # Mean Percentage Error
        ('MPE', lambda y, pred: ((y - pred) / y).mean()),
        # Root Mean Squared Error
        ('RMSE', lambda y, pred: metrics.mean_squared_error(y, pred)**0.5),
    )

    def __init__(self, features_csv, global_file):
        """Only csv file to be load DataFrame in worker.

        Args:
            features_csv (str): Location of the file with features.
            global_file (str): Filename to add CPU number, ending with `.csv`.
        """
        self._features_csv = features_csv
        # Output-related attributes
        self._global_file = global_file
        self._csv_gen = None
        self._writer = None
        # Evaluation-related
        self._df_full = None
        # # DataFrame with only the model's features
        self._df = None
        self._apps = None
        # # Main loop attributes
        self._model, self._set = None, None

    def evaluate(self, args):
        """Evaluate models."""
        creator, cpu, cpus = args
        self._init_worker(cpu)
        for self._model in creator.get_models(cpu, cpus):
            df_model = self._set_model()
            for app in self._apps:
                self._set_app(app, df_model)
                self._train()
                for self._set in ('profiling', 'target'):
                    idx = (app, self._set, self._model.number)
                    self._writer.writerow(idx + self._evaluate())
        self._csv_gen.close()

    def _set_model(self):
        cols = self._model.features + [self._model.y, 'set', 'application']
        return self._df_full[cols].replace([np.inf, -np.inf], np.nan).dropna()

    def _set_app(self, app, df_model):
        self._df = df_model[df_model.application == app]

    def _init_worker(self, cpu):
        """Initialize worker data."""
        np.random.RandomState(42)
        self._init_csv(cpu)
        self._df_full = pd.read_csv(self._features_csv)
        self._apps = sorted(self._df_full['application'].unique())

    def _init_csv(self, cpu):
        filename = '{}-{:02d}.csv'.format(self._global_file[:-4], cpu)
        self._csv_gen = CSVGen()
        if cpu == 0:
            header = ['application', 'set', 'model'] + \
                     [title for title, _ in self.METRICS]
        else:
            header = None
        self._writer = self._csv_gen.get_writer(header, filename)

    def _evaluate(self):
        y, pred = self._predict()
        if self._model.is_log:
            y = 2**y
        return tuple(t[1](y, pred) for t in self.METRICS)

    def _train(self):
        profiling = self._df.query('set == "profiling"')
        x = profiling[self._model.features]
        y = profiling[self._model.y]
        self._model.linear_model.fit(x, y)

    def _predict(self):
        df = self._df[self._df.set == self._set]
        x = df[self._model.features]
        predictions = self._model.predict(x)
        return df[self._model.y], predictions
