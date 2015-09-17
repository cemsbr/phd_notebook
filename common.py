# pylint: disable=unused-import
# Notebook configuration and modules import

import glob
from functools import partial

from IPython.core.display import Markdown
import matplotlib.pyplot as plt
import numpy as np

from lib import Experiment, HDFS, TaskModels, Plotter, get_polynomial, \
                get_r2_from_model, get_r2_from_poly
