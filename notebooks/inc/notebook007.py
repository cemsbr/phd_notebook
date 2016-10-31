"""Include file for Jupyter notebook number 007."""
import re

import numpy as np
import pandas as pd
from sklearn import metrics

from lib.bundler import Bundler
from lib.column_manager import ColumnManager
from lib.feature_set_chooser import FeatureSetChooser
from lib.group_outlier import GroupOutlier
from lib.group_ridge_cv import GroupRidgeCV
from lib.stage_df import StageDataFrame

Bundler.set_bundles_root('..', '..', 'bundles')

__all__ = 'Bundler', 'GroupOutlier', 'np', 'pd', 'StageDataFrame'


def set_column_manager():
    """Set ColumnManager class' attributes."""
    def input_per_worker(df):
        """Stage input size per worker."""
        col = 'i/w'
        values = df.s_in / df.workers
        return col, values

    def output_per_worker(df):
        """Stage input size per worker."""
        col = 'o/w'
        values = df.s_out / df.workers
        return col, values

    # ColumnManager.add_col_set('i/w', [input_per_worker])
    # ColumnManager.add_col_set('io/w', [input_per_worker, output_per_worker])

    def log_input(df_in):
        values = df_in.copy()
        values[values == 0] = 1
        return np.log(values)

    ColumnManager.add_col_set('log', is_log=True, new_cols=[
        lambda df: ('log_input', log_input(df.s_in)),
        lambda df: ('log_output', log_input(df.s_out)),
        lambda df: ('log_workers', np.log(df.workers))
        # lambda df: ('log_dur', np.log(df.s_dur))
    ])

    def max_input_per_worker(df):
        """Stage input size per worker."""
        col = 'max_i/w'
        tasks_per_workers = np.ceil(df.s_tasks / df.workers)
        task_input = df.s_in / df.s_tasks
        values = tasks_per_workers * task_input
        return col, values

    # ColumnManager.add_col_set('max_i/w', [max_input_per_worker])

    def input_per_worker_nln(df):
        """Stage input size per worker."""
        col = 's_i/w_nln'
        n = df.s_in / df.workers
        n[n == 0] = 1
        values = n * np.log2(n)
        return col, values

    new_cols = [input_per_worker_nln]
    # ColumnManager.add_col_set('s_i/w_nln', cols, new_cols)

    def input_per_worker_n2(df):
        """Stage input size per worker."""
        col = 's_i/w_n2'
        n = df.s_in / df.workers
        values = n**2
        return col, values

    new_cols = [input_per_worker_n2]
    # ColumnManager.add_col_set('s_i/w_n2', cols, new_cols)

    # ColumnManager.add_col_set('dur', cols, [lambda df: ('s_dur_', df.s_dur)])

    # ColumnManager.add_col_set('magic', cols, [lambda df: ('s_magic',
    #                                           df.s_in**1.5 / df.s_tasks)])


def set_class_attribs(df):
    """Set class attributes related to grouping samples."""
    groups = df.apply(lambda df: (df.workers, df.input), axis=1)
    groups.name = 'group'
    GroupOutlier.set_groups(groups)
    FeatureSetChooser.set_groups(groups)
    GroupRidgeCV.set_groups(groups)
    ColumnManager.set_y_column('s_dur')


# def remove_outliers(df):
#     """Remove lines with outlier in any stage."""
#     sdf = StageDataFrame(df)
#     is_outlier = pd.Series([False] * len(df), index=df.index)
#     for stage, stage_df in sdf.get_stages_df():
#         is_stage_outlier = GroupOutlier.is_outlier(stage_df, 's_dur')
#         print('stage', stage, 'outliers:', sum(is_stage_outlier))
#         is_outlier = is_outlier | is_stage_outlier
#     print('Total outliers: ', sum(is_outlier), 'of', len(is_outlier))
#     return df[~is_outlier]

def remove_outliers(df):
    """Remove total duration outliers."""
    is_outlier = GroupOutlier.is_outlier(df, 'duration_ms')
    print('target outliers:', sum(is_outlier), 'of', len(is_outlier))
    return df[~is_outlier]


def main():
    """Main function."""
    csv = Bundler.get_bundle('only_memory').get_file('only_memory.csv')
    df = pd.read_csv(csv)
    set_class_attribs(df)
    set_column_manager()
    for app in sorted(df.application.unique()):
        print(app)
        app_df = df[df.application == app]
        model = AppModel(app_df)
        model.fit()
        print(app, model.score()**0.5 / 1000)


class AppModel:
    """Application model based on stage durations."""

    def __init__(self, df):
        """Set the experiment results.

        Args:
            df (pd.DataFrame): Application DataFrame.
        """
        self._df = df
        #: ColumnManager: Provide all feature sets to select from.
        self._cm = ColumnManager()
        #: list: Stage models
        self._models = []
        self._lm = GroupRidgeCV(fit_intercept=True)

    def fit(self):
        """Train stages and the whole application."""
        df = self._df.query('set == "profiling"')
        self._fit_stages(df)
        self._fit_app(df)

    def _fit_stages(self, df):
        """Train all stages."""
        sdf = StageDataFrame(df)
        stages_df = sdf.get_stages_df(remove_outliers=True)
        self._models = [self._fit_stage(stage, df) for stage, df in stages_df]

    def _fit_stage(self, stage, df):
        """Fit a single stage.

        Args:
            stage (int): Stage number.
            df (pd.DataFrame): Stage DataFrame.
        """
        model = StageModel()
        self._cm.set_source_df(df)
        # print('fitting stage', stage)
        model.fit(self._cm)
        return model

    def _fit_app(self, df):
        """The stage duration sum is one part of the application model.

        Here, we train other the complete model for the application.
        """
        df = remove_outliers(df)
        dur_p = re.compile(r'^s\d+_dur$')
        s_dur = [c for c in df if dur_p.match(c)]

        x = pd.DataFrame(index=df.index)
        x['stages'] = df[s_dur].sum(axis=1)
        x['workers'] = df.workers

        y = df.duration_ms
        self._lm.fit(x, y)
        print('app coefs:', self._lm._ridge.intercept_, self._lm._ridge.coef_)

    def predict(self, x=None):
        if x is None:
            x = remove_outliers(self._df.query('set == "target"'))
        stages = self._predict_stages(x)
        return self._predict_app(x, stages)

    def _predict_app(self, x, stages):
        app_x = stages.copy()
        app_x['workers'] = x.workers
        return self._lm.predict(app_x)

    def _predict_stages(self, x):
        """Sum the stages' duration predictions."""
        sdf = StageDataFrame(x)
        series = sum(self._predict_stage(stage, df)
                     for stage, df in sdf.get_stages_df(remove_outliers=False))
        return pd.DataFrame(series, index=x.index)

    def _predict_stage(self, stage, df):
        model = self._models[stage]
        # print(stage, model._set)
        self._cm.set_source_df(df)
        return model.predict(self._cm)

    def score(self):
        """Return the error metric (MSE)."""
        x = remove_outliers(self._df.query('set == "target"'))
        y_pred = self.predict(x)
        y_true = x[['duration_ms']]
        return metrics.mean_squared_error(y_true, y_pred)


class StageModel:
    """Model for one application stage."""

    def __init__(self):
        """Initialize instance attributes."""
        #: str: Chosen column set by training.
        self._set = None
        #: GroupRidgeCV: Trained GroupRidgeCV instance.
        self._lm = None

    def fit(self, col_mgr):
        """Train the model.

        Args:
            col_mgr (ColumnManager): Provide all feature sets to select from.
        """
        fc = FeatureSetChooser()
        col_sets = col_mgr.get_col_sets()
        self._set, self._lm = fc.choose_feature_set(col_sets)

    def predict_score(self, col_mgr):
        """Return prediction and score."""
        x, y_true = col_mgr.get_xy(self._set)
        y_pred = self._lm.predict(x)
        score = metrics.mean_squared_error(y_true, y_pred)
        return y_pred, score

    def predict(self, col_mgr):
        """Predict from the source DataFrame of col_mgr.

        Args:
            col_mgr (ColumnManager): Column manager with the source DataFrame
                to predict.
        """
        x = col_mgr.get_x(self._set)
        y = self._lm.predict(x)
        return np.exp(y) if col_mgr.is_log(self._set) else y

    def score(self, col_mgr):
        """Predict and return the error.

        Args:
            col_mgr (ColumnManager): Column manager with the source DataFrame
                to predict. Also contains prediction target true values.

        Returns:
            float: Error metric (mean squared error).
        """
        y_pred = self.predict(col_mgr)
        y_true = col_mgr.get_y(self._set)
        return metrics.mean_squared_error(y_true, y_pred)


if __name__ == '__main__':
    main()
