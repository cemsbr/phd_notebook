"""Include file for Jupyter notebook number 007."""
import logging
import re

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import (GroupKFold, LeaveOneGroupOut,
                                     cross_val_predict)
from sklearn.linear_model import LinearRegression, RidgeCV

from lib.bundler import Bundler
from lib.column_manager import ColumnManager
from lib.feature_set_chooser import FeatureSetChooser
from lib.group_outlier import GroupOutlier
from lib.group_ridge_cv import GroupRidgeCV
from lib.stage_df import StageDataFrame

__all__ = 'Bundler', 'GroupOutlier', 'np', 'pd', 'StageDataFrame'

Bundler.set_bundles_root('..', '..', 'bundles')
fmt = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(format=fmt, level=logging.INFO, datefmt='%H:%M:%S')
log = logging.getLogger()


def set_column_manager():
    """Set ColumnManager class' attributes."""

    # # K-means
    # ColumnManager.add_col_set('kmeans', [
    #     # lambda df: ('input', df.s_in),  # 1: 244
    #     lambda df: ('output', df.s_out),  # 1: 33, 2: 12
    #     # lambda df: ('workers', df.workers),  # 1: 30
    #     lambda df: ('input/workers', df.s_in / df.workers),  # 1: 23
    #     # lambda df: ('output/workers', df.s_out / df.workers)  # 1: 39
    # ])

    # # All
    # ColumnManager.add_col_set('kmeans', [
    #     lambda df: ('input', df.s_in),  # 1: 244
    #     lambda df: ('output', df.s_out),  # 1: 33, 2: 12
    #     lambda df: ('workers', df.workers),  # 1: 30
    #     lambda df: ('1/workers', 1 / df.workers),  # 1: 23
    #     lambda df: ('input/workers', df.s_in / df.workers),  # 1: 23
    #     lambda df: ('output/workers', df.s_out / df.workers)  # 1: 39
    # ])

    # # Log
    # ColumnManager.add_col_set('logs i,o,w', use_logs=True, new_cols=[
    #     lambda df: ('log_input', df.input),
    #     lambda df: ('log_output', df.s_out),
    #     lambda df: ('log_workers', df.workers)
    #     # lambda df: ('log_dur', np.log(df.s_dur))
    # ])

    ColumnManager.add_col_set('i/w', [
        lambda df: ('i/w', df.input/df.workers),
        lambda df: ('o/w', df.s_out/df.workers),
        lambda df: ('i', df.s_in/df.workers),
        lambda df: ('w', df.workers)
    ])

    # # Wikipedia and HBSort (ranks for Wikipedia)
    # ColumnManager.add_col_set('wiki_sort', [
    #     lambda df: ('input', df.input),  # 2
    #     # lambda df: ('output', df.s_out),
    #     # lambda df: ('workers', df.workers),
    #     lambda df: ('input/workers', df.input / df.workers),  # 1
    #     # lambda df: ('output/workers', df.s_out / df.workers)
    # ])


def set_class_attribs(df):
    """Set class attributes related to grouping samples."""
    # groups = df.apply(lambda df: (df.workers, df.input), axis=1)
    GroupOutlier.set_groups(df[['workers', 'input']])
    ColumnManager.set_y_column('s_dur')


def remove_outliers(df):
    """Remove total duration outliers."""
    is_outlier = GroupOutlier.is_outlier(df, 'duration_ms')
    log.info('  outliers in total duration: %d of %d', sum(is_outlier),
             len(is_outlier))
    return df[~is_outlier]


def format_units(df):
    time_cols = [c for c in df if c.endswith('_dur')] + ['duration_ms']
    bytes_cols = [c for c in df if c.endswith('_in') or c.endswith('_out')] + \
        ['input']
    df[time_cols] /= 1000  # seconds
    df[bytes_cols] /= 1024**2  # Megabytes


def read_csv():
    csv = Bundler.get_bundle('only_memory').get_file('only_memory.csv')
    df = pd.read_csv(csv)
    format_units(df)
    return df


def main(df):
    """Main function."""
    set_class_attribs(df)
    set_column_manager()
    # for app in ['hbsort', 'wikipedia']:
    for app in sorted(df.application.unique()):  # noqa
        log.info(app.upper())
        log.info('=' * len(app))
        app_df = df[df.application == app]
        model = AppModel(app_df)
        model.fit()
        log.info('%s RMSE: %.4f', app, model.score()**0.5)


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
        self._models_dur = []
        self._models_out = []
        self._delay = None
        self._cv = LeaveOneGroupOut()

    def fit(self):
        """Train stages and the whole application."""
        log.info('Fitting...')
        df = remove_outliers(self._df.query('set == "profiling"'))
        self._fit_delay(df)
        self._fit_stages(df)

    def predict(self, df=None):
        log.info('Predicting...')
        if df is None:
            df = remove_outliers(self._df.query('set == "target"'))
        delay = self._predict_delay(df)
        stages = self._predict_stages(df)
        return delay + stages

    def _fit_delay(self, df):
        # x = pd.DataFrame(np.ones(len(df)), index=df.index)
        x = df[['workers']]
        # x = df[['input']]
        # x = df[['workers', 'input']]
        sdf = StageDataFrame(df)
        # Curious: can we sum all s\d\d\_dur cols, even nan ones?
        stages_dur = sum(s_df.s_dur for _, s_df in sdf.get_stages_df(False))
        y_true = df.duration_ms - stages_dur

        # groups = df.apply(lambda df: '{:03d}, {:.9f}'.format(df.workers,
        #                   df.input), axis=1)
        groups = df.workers
        # groups = df.input
        self._delay = GroupRidgeCV(groups, LeaveOneGroupOut(),
                                   fit_intercept=True)
        mse, mse_log = self._delay.fit(x, y_true)
        log.info('  Delay RMSE = %.4f sec', mse**0.5)
        log.debug('  Delay RMSE (log) = %.4f sec', mse_log**0.5)
        ridge = self._delay._ridge
        log.debug('  Delay coefs = %s %.4f, alpha = %s', ridge.coef_,
                  ridge.intercept_, ridge.alpha)

    def _predict_delay(self, df):
        # x = pd.DataFrame(np.ones(len(df)), index=df.index)
        x = df[['workers']]
        # x = df[['input']]
        # x = df[['workers', 'input']]
        y_pred = self._delay.predict(x)

        sdf = StageDataFrame(df)
        stages_dur = sum(s_df.s_dur for _, s_df in sdf.get_stages_df(False))
        y_true = df.duration_ms - stages_dur
        mse = metrics.mean_squared_error(y_true, y_pred)
        log.info('  Delay RMSE = %.4f sec', mse**0.5)

        return y_pred

    def _fit_stages(self, df):
        """Train all stages."""
        sdf = StageDataFrame(df)
        stages_df = sdf.get_stages_df(remove_outliers=True)
        scr_dur_sum, scr_out_sum = 0, 0
        assert df.input.equals(df.s00_in)
        for _, df in stages_df:
            scr_out, scr_dur = self._fit_stage(df)
            scr_out_sum += scr_out
            scr_dur_sum += scr_dur
        log.info("  Sum of stages' errors:")
        log.info('    Output  = %.4f MB', scr_out_sum)
        log.info('    Duration = %.4f sec', scr_dur_sum**0.5)

    def _predict_stages(self, df):
        sdf = StageDataFrame(df)
        stages_df = sdf.get_stages_df(remove_outliers=False)
        scr_dur_sum, scr_out_sum = 0, 0
        s_out = df.input
        pred_sum = 0
        for stage, s_df in stages_df:
            col_preffix = 's{:02d}_'.format(stage)
            df[col_preffix + 'in'] = s_out
            pred, scr_dur, scr_out, s_out = self._predict_stage(s_df, stage)
            df[col_preffix + 'out'] = s_out
            pred_sum += pred
            scr_dur_sum += scr_dur
            scr_out_sum += scr_out

        log.info("  Sum of stages' errors:")
        log.info('    Output   = %.4f MB', scr_out_sum)
        log.info('    Duration = %.4f sec', scr_dur_sum**0.5)

        return pred_sum

    def _fit_stage(self, df):
        """Fit a single stage.

        Args:
            df (pd.DataFrame): Stage DataFrame.

        Returns:
            float: Duration cross-validation training score, in milliseconds.
            float: Output cross-validation training score, in gigabytes.
        """
        return self._fit_stage_output(df), self._fit_stage_dur(df)

    def _predict_stage(self, df, stage):
        score_out = self._predict_stage_output(df, stage)
        y_pred, score_dur = self._predict_stage_dur(df, stage)
        return y_pred, score_dur, score_out, df.s_out

    def _fit_stage_output(self, df):
        x = df[['input', 'workers']]
        y_true = df.s_out

        groups = df.apply(lambda df: '{:03d}, {:.8f}'.format(df.workers,
                          df.input), axis=1)
        # groups = df.input
        if df.application.iloc[0] == 'hbkmeans':
            cv = GroupKFold(n_splits=3)
        else:
            cv = LeaveOneGroupOut()
        lm = GroupRidgeCV(groups, cv)
        # lm = LinearRegression()
        self._models_out.append(lm)
        # y_pred = cross_val_predict(lm, x, y_true, groups, LeaveOneGroupOut())
        # lm.fit(x, y_true)
        # y_pred = lm.predict(x)
        # return metrics.mean_squared_error(y_true, y_pred)
        return lm.fit(x, y_true)[0]

    def _predict_stage_output(self, df, stage):
        x = df[['input', 'workers']]
        y_true = df.s_out
        lm = self._models_out[stage]
        y_pred = lm.predict(x)
        mse = metrics.mean_squared_error(y_true, y_pred)
        df['s_out'] = y_pred
        return mse

    def _fit_stage_dur(self, df):
        model = StageModel()
        self._cm.set_source_df(df)
        score = model.fit(self._cm)
        self._models_dur.append(model)
        return score

    def _predict_stage_dur(self, df, stage):
        model = self._models_dur[stage]
        self._cm.set_source_df(df)
        y_pred = model.predict(self._cm)
        score = model.score(self._cm, y_pred)
        return y_pred, score

    def score(self):
        """Return the error metric (MSE)."""
        df = remove_outliers(self._df.query('set == "target"'))
        y_pred = self.predict(df)
        y_true = df[['duration_ms']]
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

        Returns:
            float: Column set cross validation training score.
        """
        groups = global_df.apply(lambda df: '{:03d}, {:.8f}'.format(
            df.workers, df.input), axis=1)
        if col_mgr._src.application.iloc[0] == 'hbkmeans':
            cv = GroupKFold(n_splits=3)
        else:
            cv = LeaveOneGroupOut()
        fc = FeatureSetChooser(groups, cv)
        col_sets = col_mgr.get_col_sets()
        score, self._set, self._lm = fc.choose_feature_set(col_sets)
        return score

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
        return self._lm.predict(x)
        # return np.exp(y) if col_mgr.is_log(self._set) else y

    def score(self, col_mgr, y_pred=None):
        """Predict and return the error.

        Args:
            col_mgr (ColumnManager): Column manager with the source DataFrame
                to predict. Also contains prediction target true values.

        Returns:
            float: Error metric (mean squared error).
        """
        if y_pred is None:
            y_pred = self.predict(col_mgr)
        y_true = col_mgr.get_y(self._set)
        return metrics.mean_squared_error(y_true, y_pred)


if __name__ == '__main__':
    global_df = read_csv()
    main(global_df)
