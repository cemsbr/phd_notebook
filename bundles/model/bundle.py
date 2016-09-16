"""Evaluate models."""
import multiprocessing
import pickle

import pandas as pd

from bundles.common_parsing import CSVGen
from lib.bundler import BaseBundle
from lib.config import Config
from lib.model_creator import ModelCreator
from lib.model_evaluator import ModelEvaluator


class Bundle(BaseBundle):
    """Evaluate models."""

    def __init__(self):
        """File to be created."""
        super().__init__('evaluation.csv', 'models.csv')
        self.dependencies = 'features',

    def run(self):
        """Generate and evaluate all models."""
        self.start()

        # Getting all the features for regular (not log) regression.
        features_csv = self.bundler.get_bundle('features').filename
        exclude_cols = ['application', 'set', Config.Y, Config.LOG_Y]
        all_cols = pd.read_csv(features_csv).columns.tolist()
        cols = [c for c in all_cols if c not in exclude_cols]

        creator = ModelCreator(Config.LINEAR_MODELS, cols, Config.Y,
                               Config.LOG_FEATURES, Config.LOG_Y)
        self.log.info('%d models to evaluate...', len(creator))
        # Models generator
        cpus = multiprocessing.cpu_count()
        models = [(creator, cpu, cpus) for cpu in range(cpus)]

        self._dump_models(creator)
        self._evaluate(models, features_csv)

        self.finish()

    def _dump_models(self, creator):
        csv_gen = CSVGen()
        writer = csv_gen.get_writer(('number', 'dump'), self.filenames[1])
        for model in creator.get_models(0, 1):
            writer.writerow((model.number, pickle.dumps(model)))
        csv_gen.close()
        self.log.info('Models dumped.')

    def _evaluate(self, models, features_csv):
        csv_gen = CSVGen()
        header = ModelEvaluator.get_csv_header()
        evaluator = ModelEvaluator(features_csv)
        writer = csv_gen.get_writer(header, self.filenames[0])
        with multiprocessing.Pool() as p:
            for rows in p.map(evaluator.evaluate, models):
                writer.writerows(rows)
        csv_gen.close()


if __name__ == '__main__':
    Bundle().update()
