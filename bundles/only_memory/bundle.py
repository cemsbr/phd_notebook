"""Prediction for when data does not fit the memory."""
import pandas as pd

from lib.bundler import BaseBundle
from lib.memory_predictor import MemoryPredictor, Report


class Bundle(BaseBundle):
    """Prediction and comparison of when memory is not enough."""

    def __init__(self):
        """`filenames` and `dependencies` setting."""
        super().__init__('only_memory.csv')
        self.dependencies = ['all_apps']

    def run(self):
        """Remove cases when memory is not enough.

        In the profiling set, use true information. For the target, use
        prediction.
        """
        self.start()
        csv_file = self.bundler.get_bundle(self.dependencies[0]).filename
        df = pd.read_csv(csv_file)

        profiling = df[df.set == 'profiling']
        target = df[df.set == 'target']

        dfs = []
        # profiling - only in_memory, remove in_memory column
        dfs.append(profiling[profiling.in_memory].drop('in_memory', axis=1))

        # target - remove not in_memory according to prediction
        pred = MemoryPredictor(profiling)
        dfs.append(pred.filter_in_memory(target))

        pd.concat(dfs).to_csv(self.filename, index=False)

        report = Report(pred)
        self.log.info(report.get_text('profiling set', profiling))
        self.log.info(report.get_text('target set', target))
        self.finish()


if __name__ == '__main__':
    Bundle().update()
