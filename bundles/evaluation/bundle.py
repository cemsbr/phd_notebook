"""Evaluate models."""
import multiprocessing

import pandas as pd

from lib.bundler import BaseBundle
from lib.config import Config
from lib.model_creator import ModelCreator
from lib.model_evaluator import ModelEvaluator


class Bundle(BaseBundle):
    """Evaluate models."""

    def __init__(self):
        """File to be created."""
        super().__init__('evaluation.csv', 'models.json')
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
        with open(self.filenames[1], 'w') as f:
            for model in creator.get_models(0, 1):
                f.write(model.to_json() + '\n')
        self.log.info('Models dumped.')

    def _evaluate(self, models, features_csv):
        evaluator = ModelEvaluator(features_csv, self.filenames[0])
        with multiprocessing.Pool() as p:
            p.map(evaluator.evaluate, models, chunksize=1)


if __name__ == '__main__':
    Bundle().update()
