"""Code used in notebook 1."""
import pickle

from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import RidgeCV

import inc.helper as helper
import inc.pd_config  # pylint: disable=W0611
import inc.plt_config  # pylint: disable=W0611
from lib import Humanizer, Plotter
from lib.hbkmeans import DataFrameBuilder as DFBuilderK
from lib.hbsort import DataFrameBuilder as DFBuilderSort
from lib.wikipedia import DataFrameBuilder as DFBuilderWiki

__all__ = ('DFBuilderK', 'DFBuilderSort', 'DFBuilderWiki', 'display', 'helper',
           'Humanizer', 'linear_model', 'np', 'pd', 'pickle', 'Plotter', 'plt',
           'RidgeCV')
