"""Jupyter notebook display configuration."""
# pylint: disable=W0611
import matplotlib
import pandas as pd
from IPython.display import display
from lib import Outlier


matplotlib.style.use('ggplot')  # R style
matplotlib.rcParams.update({'font.size': 16,
                            'font.family': 'serif',
                            'lines.linewidth': 2,
                            'axes.xmargin': 0.1,
                            'axes.ymargin': 0.1})
pd.options.display.precision = 2


def process_outliers(df, humanizer, plotter, caption):
    """Add outlier column and return only non-outliers with no extra column."""
    df['outlier'] = Outlier.is_outlier(df)
    overview = Outlier.get_overview(df)
    display(humanizer.humanize(overview))
    plotter.plot_outliers(df)
    print(caption)
    return df[~df.outlier].drop('outlier', axis=1)

def remove_outliers(df):
    outliers = Outlier.is_outlier(df)
    return df[~outliers]
