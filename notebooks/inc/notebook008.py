"""Trying to get the same results of manual approach by using ML."""
import logging
from functools import partial

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from lib.bundler import Bundler


# Configuration
# fmt = '%(asctime)s %(levelname)s %(message)s'
fmt = '%(levelname)s %(message)s'
logging.basicConfig(format=fmt, level=logging.INFO, datefmt='%H:%M:%S')
log = logging.getLogger()
log.info('Started')
Bundler.set_bundles_root('..', '..', 'bundles')
pd.set_option('display.precision', 2)


def change_units(df, time_cols, data_cols, task_cols):
    """Make the numbers smaller."""
    df[time_cols] /= 1000  # to seconds
    df[data_cols] /= 1024**2  # to Megabytes
    for task_col in task_cols:
        df[task_col] = df[task_col].astype(int)


def detect_outliers(df, group_cols, col, m=1.5):
    """Return an array with boolean True for outliers."""
    def iqr(values):
        """Return the interquartile range."""
        return values.quantile(q=.75) - values.quantile(q=.25)

    group = df.groupby(group_cols)[col]
    lower = group.transform(lambda df: df.quantile(q=.25) - m * iqr(df))
    upper = group.transform(lambda df: df.quantile(q=.75) + m * iqr(df))
    return (df[col] < lower) | (df[col] > upper)


# Common variables
csv = Bundler.get_bundle('only_memory').get_file('only_memory.csv')
df_all = pd.read_csv(csv)


def get_pred(lm, x, y_true, use_logs):
    if use_logs:
        lm.fit(np.log(x), np.log(y_true))
        y_pred = np.exp(lm.predict(np.log(x)))
    else:
        lm.fit(x, y_true)
        y_pred = lm.predict(x)
    return y_pred


def get_pred_cv(lm, x, y_true, groups, use_logs):
    cv = LeaveOneGroupOut()
    y_pred = pd.Series()
    for train_ix, test_ix in cv.split(x, y_true, groups):
        x_train = x.iloc[train_ix]
        y_train = y_true.iloc[train_ix]
        x_test = x.iloc[test_ix]
        lm.fit(x_train, y_train)
        if use_logs:
            lm.fit(np.log(x_train), np.log(y_train))
            arr = np.exp(lm.predict(np.log(x_test)))
        else:
            lm.fit(x_train, y_train)
            arr = lm.predict(x_test)
        s = pd.Series(arr, index=x_test.index)
        y_pred = y_pred.append(s, verify_integrity=True)
    return y_pred.sort_index()


def eval_model(df, title, x_fn, time_col, use_logs, cv=True):
    """Calculate errors using all x and y."""
    x = x_fn(df)
    y_true = df[time_col]
    lm = LinearRegression()
    y_pred = get_pred_cv(lm, x, y_true, df.input, use_logs) if cv else \
        get_pred(lm, x, y_true, use_logs)
    df['pred'] = y_pred
    df['abs_diff'] = np.abs(df[time_col] - y_pred)
    log.debug(title)
    log.debug('\n%s', df)
    # log.debug('coefs: %.4f + %s', lm.intercept_, lm.coef_)
    rmse = mean_squared_error(y_true, y_pred)**0.5
    r2 = r2_score(y_true, y_pred)
    max_diff = df.abs_diff.max()
    log.debug('  rmse = %.4f sec, r2 = %.4f, max diff = %.2f.', rmse, r2,
              max_diff)
    lm.fit(x, y_true)
    return (rmse, r2, max_diff, lm)


def evaluate(df, app, stage):
    """Evaluate a stage of an app."""
    # Load data
    df = df[df.application == app].copy()
    df.dropna(thresh=len(df), axis=1, inplace=True)
    df.drop('application', axis=1, inplace=True)

    # Common variables
    time_cols = [c for c in df if c.endswith('_dur')] + ['duration_ms']
    data_cols = [c for c in df if c.endswith('_in') or c.endswith('_out')] + \
                ['input']
    task_cols = [c for c in df if c.endswith('_tasks')]

    change_units(df, time_cols, data_cols, task_cols)
    # How many stages?
    s_dur_cols = [col for col in df if col.endswith('_dur')]
    log.info('app %s, stage %d of %d.', app, stage, len(s_dur_cols))

    # Average stage time by input and workers
    task_col = task_cols[stage]
    time_col = time_cols[stage]
    mean = df[['workers', task_col, 'input', time_col]]
    is_outlier = detect_outliers(mean, ['workers', 'input'], time_col)
    log.info('%d outliers out of %d.', sum(is_outlier), len(is_outlier))
    mean = mean[~is_outlier]
    # mean = mean.groupby(['workers', 'input']).mean().reset_index()

    models = []

    def x_input_per_worker(df_):
        x = df_[['input']].copy()
        x['input'] /= df_.workers
        return x

    def x_tasks_per_worker(df_):
        x = df_[[task_col]].copy()
        x[task_col] /= df_.workers
        return np.ceil(x)

    def x_log_workers(df_):
        return np.log(df_[['workers']])

    def x_input_workers(df_):
        return df_[['input', 'workers']]

    def x_tasks_workers(df_):
        return df_[[task_col, 'workers']]

    def x_magic(k, df_):
        x = df_[['input']].copy()
        x.input /= np.log(df_.workers + k)  # 0.45 for stage 1
        return x

    models.append(('Input/Workers', x_input_per_worker, False))
    models.append(('Tasks/Workers', x_tasks_per_worker, False))
    models.append(('log(Workers) (kmeans 14, 16, 18)', x_log_workers, False))
    models.append(('Log: Input, Workers', x_input_workers, True))
    models.append(('Log: Tasks, Workers', x_tasks_workers, True))
    models.append(('hbsort 1', partial(x_magic, 0.3182), False))
    models.append(('hbsort 0', partial(x_magic, 0.45), False))

    results = []
    for model in models:
        title, x_fn, use_logs = model
        result = eval_model(mean, title, x_fn, time_col, use_logs)
        results.append(result + (title,))
    results.sort()
    log.info('Best for %s, stage %s is %s.', app, stage, results[0][-1])
    log.debug(results[0][:-2])
    return results


def evaluate_all(df):
    """Evaluate all apps, all stages."""
    apps = 'hbkmeans', 'hbsort', 'wikipedia'
    stage_totals = 25, 2, 3
    results = []
    for app, total_stages in zip(apps, stage_totals):
        for stage in range(total_stages):
            results.append(evaluate(df, app, stage))
    return results


def predict(df, lms):
    apps = 'hbkmeans', 'hbsort', 'wikipedia'
    stage_totals = 25, 2, 3
    for app, total_stages in zip(apps, stage_totals):
        for stage in range(total_stages):
            predict_stage(df, app, stage)


def main(df):
    results = evaluate_all(df_all.query('set == "profiling"'))
    lms = (res[-1] for res in results)
    # predict(df_all.query('set == "target"'), lms)


if __name__ == '__main__':
    main(df_all)


log.info('Finished\n')

# Notes:
# Some stages has a fixed number of tasks, so input per worker must be used.
# Wikipedia is a little better with ceil(tasks/workers). In all stages,
# input/workers works best.
# - Stage 0: ceil(tasks/workers), input/workers
# - Stage 1: input/workers
# - Stage 2: log: input, workers
# HBSort:
# - Stage 0: log(input/workers)
