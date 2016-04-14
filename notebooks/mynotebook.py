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


def process_outliers(df, humanizer, caption=None, plotter=None):
    """Add outlier column and return only non-outliers with no extra column."""
    outlier = Outlier(df)
    overview = outlier.get_overview()
    display(humanizer.humanize(overview))
    if plotter:
        plotter.plot_outliers(df)
    if caption:
        print(caption)
    return outlier.remove()


def remove_outliers(df):
    outliers = Outlier.detect(df)
    return df[~outliers]
