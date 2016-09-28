"""Shorten notebook code."""
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from IPython.display import display, Markdown

# pylint: disable=W0611
import inc.pd_config
import inc.plt_config  # noqa
from lib.bundler import Bundler
from lib.config import Config
from lib.hbkmeans_parser import HBKmeansParser
from lib.hbsort_parser import HBSortParser
from lib.memory_predictor import MemoryPredictor, Report
from lib.model_creator import Model, ModelCreator
from lib.plotter import Plotter
from lib.wikipedia_parser import WikipediaParser


Bundler.set_bundles_root('..', '..', 'bundles')

__all__ = ('add_ranks', 'Bundler', 'display', 'find_model', 'get_model',
           'get_model_df', 'get_model_creator', 'get_summary', 'Markdown',
           'Model', 'np', 'pd', 'plot_model', 'predict_memory', 'stage_tasks',
           'select_best', 'tasks_blocks', 'plot_wikipedia_cost')


def predict_memory():
    """Predict when there is an increased disk usage."""
    # Loading profiling and target datasets
    csv_file = Bundler.get_bundle('all_apps').filename
    df = pd.read_csv(csv_file)
    profiling = df.query('set == "profiling"')
    target = df.query('set == "target"')

    # Prediction and accuracy
    predictor = MemoryPredictor(profiling)
    report = Report(predictor)
    print(report.get_text('profiling set', profiling))
    print(report.get_text('target set', target))


def get_summary():
    """Summary with number of outliers and average execution time.

    Execution time only for non-outliers.
    """
    df = pd.read_csv(Bundler.get_bundle('outlier_detection').get_file(
        'outliers_detected.csv'))
    df['total'] = 1
    df.sort_values('input', inplace=True)

    group_cols = ['application', 'set', 'input', 'workers']
    group = df.drop('duration_ms', axis=1).groupby(group_cols)
    s = group.agg({'total': sum, 'outlier': sum})
    # Exclude outliers for execution mean
    s['duration_ms'] = df[~df.outlier].groupby(group_cols). \
        duration_ms.mean() / 1000
    s.rename(columns={'outlier': 'outliers', 'duration_ms': 'mean (sec)'},
             inplace=True)
    s.outliers = s.outliers.astype(int)
    s['available'] = s.total - s.outliers
    return s[['mean (sec)', 'available', 'outliers', 'total']]


def stage_tasks():
    """Display tasks per stage and count their occurencies."""
    print('Wikipedia:')
    display(_app_stage_tasks(t[0] for t in WikipediaParser().get_apps()))
    print('HBKmeans:')
    display(_app_stage_tasks(HBKmeansParser.get_apps()))
    print('HBSort:')
    display(_app_stage_tasks(HBSortParser.get_apps()))


def tasks_blocks():
    """Compare number of tasks and number of HDFS blocks."""
    csv_file = Bundler.get_bundle('input_tasks').get_file()
    df = pd.read_csv(csv_file)
    df['HDFS Blocks'] = np.ceil(df['Size (MB)'] / 128).astype('int')
    cols = df.columns.tolist()
    df['count'] = 1
    return df.groupby(cols).sum()


def select_best(cmp_df, model_info, sset='target'):
    """Select best model of each column in cmp_df."""
    all_best = []
    for metric in cmp_df.columns:
        best = cmp_df.xs(sset, level='set').sort_values(metric).index. \
            get_level_values('model')[0]
        print('Best in "{}": {}'.format(metric, best))
        all_best.append(best)
    return Counter(all_best).most_common()


def _app_stage_tasks(apps):
    rows = ((sum(1 for _ in s.successful_tasks) for s in app.stages)
            for app in apps)
    df = pd.DataFrame(rows)
    df.rename(columns='stg {}'.format, inplace=True)
    cols = df.columns.tolist()
    df['count'] = 1
    return df.groupby(cols).sum()


def add_ranks(df):
    """Add ranks for MAPE, RMSE and rank sum to `df`."""
    for err in ('MAPE', 'RMSE'):
        for app in ('wikipedia', 'hbsort', 'hbkmeans'):
            for sset in ('profiling', 'target'):
                s = df.xs((sset, app), level=('set', 'application'))[err]
                ranks = s.sort_values().reset_index().sort_values('model'). \
                    index.values
                df.loc[(sset, slice(None), app), err + ' rank'] = ranks
    # Not sure why, but ranks are floats. Turn them into ints:
    for col in ('MAPE rank', 'RMSE rank'):
        df[col] = df[col].astype(int)
        df['rank sum'] = df['MAPE rank'] + df['RMSE rank']
    # Change col order
    df = df[['MAPE rank', 'MAPE', 'RMSE rank', 'RMSE', 'MPE', 'rank sum']]


def plot_model(model, save_pdf=False):
    """Plot all 3 applications' predictions.

    Assumes Config and features.csv are the same.

    Args:
        model: Model's number or object
        save_pdf (boolean): Whether to save the plots as PDF in /tmp/.
    """
    model = model if type(model).__name__ == "Model" else get_model(model)
    model_df = get_model_df(model)
    plot_wikipedia(model, model_df, save_pdf)
    plot_hbsort(model, model_df, save_pdf)
    plot_hbkmeans(model, model_df, save_pdf)


def _select_df(df, col, value):
    return df[df[col] == value].drop(col, axis=1)


def plot_wikipedia(model, model_df, save_pdf=False):
    """Plot actual results and model's predictions."""
    output = '/tmp/wikipedia.pdf' if save_pdf else None
    df = _select_df(model_df, 'application', 'wikipedia')
    target = _train(model, df)
    plotter = Plotter(xlim=(3, 65), ylim=(0, 200))
    plotter.plot_model(model, target, output)
    print('Prediction of the Wikipedia application target execution duration.')


def plot_wikipedia_cost(model, model_df, cost_func, save_pdf=False):
    output = '/tmp/wikipedia_cost.pdf' if save_pdf else None
    df = _select_df(model_df, 'application', 'wikipedia')
    target = _train(model, df)
    plotter = Plotter(xlim=(0, 3.5))
    plotter.update_config(s=40)
    plotter.plot_cost_model(model, target, cost_func, output)
    print('Prediction of the Wikipedia application cost.')


def plot_hbsort(model, model_df, save_pdf=False):
    """Plot actual results and model's predictions."""
    outputs = _get_outputs(save_pdf, 'hbsort3.pdf', 'hbsort30.pdf')
    df = _select_df(model_df, 'application', 'hbsort')
    target = _train(model, df)
    plotter = Plotter(xlim=(0.75, 16.25))
    plotter.plot_model(model, target[target.input < 15 * 1024**3], outputs[0])
    plotter = Plotter(xlim=(14, 130))
    plotter.plot_model(model, target[target.input > 15 * 1024**3], outputs[1])
    print('The top figure is the result of the HiBench Sort application with'
          ' 3-GB input. The second figure uses 31-GB of data.')


def plot_hbkmeans(model, model_df, save_pdf=False):
    """Plot actual results and model's predictions."""
    outputs = _get_outputs(save_pdf, 'hbkmeans16.eps', 'hbkmeans65.eps')
    df = _select_df(model_df, 'application', 'hbkmeans')
    target = _train(model, df)
    plotter = Plotter(xlim=(7.5, 32.5))
    plotter.plot_model(model, target[target.input == 16384000], outputs[0])
    plotter = Plotter(xlim=(30, 130))
    plotter.plot_model(model, target[target.input == 65536000], outputs[1])
    print('The top figure is the result of the HiBench K-means application'
          ' with 16,384,000 samples. The second figure uses 65,536,000'
          ' samples.')


def _get_outputs(save_pdf, *basenames):
    return ['/tmp/' + f for f in basenames] if save_pdf else (None, None)


def _train(model, df):
    model.train(_select_df(df, 'set', 'profiling'))
    return _select_df(df, 'set', 'target')


def get_model_creator():
    """Get model creator with values from :class:`.Config`."""
    exclude_cols = ['application', 'set', Config.Y, Config.LOG_Y]
    features_csv = Bundler.get_bundle('features').filename
    all_cols = pd.read_csv(features_csv).columns.tolist()
    cols = [c for c in all_cols if c not in exclude_cols]
    return ModelCreator(Config.LINEAR_MODELS, cols, Config.Y,
                        Config.LOG_FEATURES, Config.LOG_Y)


def get_model(number):
    """Get model from CSV file."""
    return _get_csv_model(number)


def _get_mc_model(number):
    """Get a model using :class:`.ModelCreator`.

    It is much faster, but :class:`.Config` and features' CSV must be the same.
    """
    mc = get_model_creator()
    model = mc.get_model(number)
    assert number == model.number
    return model


def _get_csv_model(number):
    """Get model from the CSV file."""
    csv_file = Bundler.get_bundle('evaluation').filenames[1] + '.bz2'
    dump_str = pd.read_csv(csv_file, skiprows=number, nrows=1, usecols=[1]). \
        ix[0][0]
    dump = eval(dump_str)  # pylint: disable=W0123
    model = pickle.loads(dump)
    assert number == model.number
    return model


def get_model_df(model):
    """Get features from CSV."""
    csv_file = Bundler.get_bundle('features').filename
    cols = model.features + [model.y, 'application', 'set', 'workers', 'input',
                             'duration_ms']
    cols = list(set(cols))
    return pd.read_csv(csv_file, usecols=cols)


def find_model(features, lm_class):
    """Find a model by features and linear model."""
    mc_model = _find_mc_model(features, lm_class)
    csv_model = _get_csv_model(mc_model.number)
    if mc_model != csv_model:
        print('Warning: Config class or features.csv differs (slower).')
        mc_model = _find_csv_model(features, lm_class)
    return csv_model


def _find_mc_model(features, lm_class):
    """Find a model by features and linear model.

    It is much faster, but :class:`.Config` and features' CSV must be the same.
    """
    feat_set = set(features)
    for model in get_model_creator().get_models(0, 1):
        model_feat = set(model.features)
        lm = model.linear_model
        if isinstance(lm, lm_class) and feat_set == model_feat:
            return model


def _find_csv_model(features, lm_class):
    """Find a model by features and linear model."""
    feat_set = set(features)
    csv_file = Bundler.get_bundle('evaluation').filenames[1] + '.bz2'
    for dump in pd.read_csv(csv_file, usecols=['dump']).dump:
        model = pickle.loads(eval(dump))  # pylint: disable=W0123
        model_feat = set(model.features)
        lm = model.linear_model
        if isinstance(lm, lm_class) and feat_set == model_feat:
            return model
