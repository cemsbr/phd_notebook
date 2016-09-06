"""Jupyter notebook display configuration."""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from lib import Outlier, FeatureCreator, ModelCreator, ModelEvaluator


# ##
# ## Begin new code for bundler
# ##

def get_outlier_overview(df_outlier):
    df = df_outlier.drop('outlier', axis=1).rename(
        columns={'duration_ms': 'samples'}).copy()
    df['outliers'] = df_outlier.outlier.astype('int')

    group_cols = ['application', 'set', 'input', 'workers']

    group = df.groupby(group_cols)
    overview = group.agg({'samples': np.size, 'outliers': np.sum})
    # Display number of available executions (non outliers)
    overview['not outliers'] = overview.samples - overview.outliers
    overview['mean (sec)'] = df[~df_outlier.outlier].groupby(
        group_cols).mean().samples/1000

    # Improving readability
    return overview[['mean (sec)', 'samples', 'outliers', 'not outliers']]

# ##
# ## End new code for bundler
# ##

# ##
# ## Get experiment data
# ##
def get_wiki_profiling_df(dfb):
    """Get Wikipedia app's profiling experiments."""
    one_vm = dfb.get_1vm()
    many_vms = dfb.get_strong_scaling(1)
    # merge both experiment sets
    return pd.concat([one_vm, many_vms[many_vms.workers > 1]])


def get_sort_target_df(dfb):
    """Get HiBench Sort target experiments."""
    exp0 = dfb.get_target(0)
    exp1 = dfb.get_target(1)
    # merge both experiment sets
    df = pd.concat([exp0, exp1])
    return df[(df.workers != 12) & (df.workers != 123)]


def remove_outliers(df, humanizer=None, caption=None, plotter=None):
    """Add outlier column and return only non-outliers with no extra column."""
    outlier = Outlier(df)
    overview = outlier.get_overview()
    if humanizer:
        display(humanizer.humanize(overview))
    else:
        display(overview)
    if plotter:
        plt.figure()
        plotter.plot_outliers(df)
    if caption:
        print(caption)
    return outlier.remove()


# ##
# ## Modeling
# ##
def get_cols(df_src, cols_fns, degree):
    ycol = 'ms'
    df = _get_cols(df_src, cols_fns)
    x, y = df.drop(ycol, axis=1), df[ycol]
    featCr = FeatureCreator(x)
    df_deg = featCr.get_poly(degree)
    df_deg[ycol] = y.values
    return df_deg


def get_log_cols(df_src, cols_fns):
    df = _get_cols(df_src, cols_fns)
    featCr = FeatureCreator(df)
    return featCr.get_log()

def evaluate_exp(fit_df,
                 score_dfs,
                 linear_models,
                 features=None,
                 is_log=False):
    ycol = 'log(ms)' if is_log else 'ms'
    if features is None:
        features = [col for col in fit_df.columns if not col == ycol]
    evaluator = _get_evaluator(fit_df, linear_models, features, ycol)
    return evaluator.evaluate(score_dfs, is_log=is_log)


def _get_cols(df_src, cols_fns):
    df = pd.DataFrame()
    for col, func in cols_fns:
        df[col] = func(df_src)
    return df


def _get_evaluator(fit_df, linear_models, features, ycol):
    models = ModelCreator(linear_models, features, ycol)
    evaluator = ModelEvaluator(models)
    evaluator.set_fit_data(fit_df)
    return evaluator


# ##
# ## Visualization
# ##
def format_results(results):
    return pd.DataFrame(
        [_format_result(res) for res in results],
        columns=['model', 'nr_feats', 'log', 'features', 'params', 'MAE',
                 'MPE', 'RMSE'])


def _format_result(result):
    errors = result['scores']
    res = _format_errors(errors)
    model = pickle.loads(result['model_dump'])
    res.update(model.humanize())
    return res


def _format_errors(errors):
    return {
        'MAE': errors['MAE'] / 1000,  # seconds
        'MPE': errors['MPE'] * 100,  # %
        'RMSE': errors['RMSE'] / 1000  # seconds
    }
