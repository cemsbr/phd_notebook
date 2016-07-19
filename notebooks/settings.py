# pylint: disable=W0611

# Python's import
import pickle

# 3rd-party libraries
from IPython.display import display
import matplotlib
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import RidgeCV

# My code
import helper
from lib import Humanizer, Model, Plotter
from lib.hbsort import DataFrameBuilder as DFBuilderSort
from lib.wikipedia import DataFrameBuilder as DFBuilderWiki
from lib.hbkmeans import DataFrameBuilder as DFBuilderK


matplotlib.style.use('ggplot')  # R style
matplotlib.rcParams.update({'font.size': 16,
                            'font.family': 'serif',
                            'lines.linewidth': 2,
                            'axes.xmargin': 0.1,
                            'axes.ymargin': 0.1})
pd.options.display.precision = 2
pd.options.display.max_columns = 99
pd.options.display.max_rows = 99
pd.options.display.float_format = lambda x: '{:.2f}'.format(x)
