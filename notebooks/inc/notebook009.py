import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from inc.plt_config import *
from lib.bundler import Bundler
from lib.wikipedia_parser import WikipediaParser

__all__ = 'Bundler', 'LinearRegression', 'mean_squared_error', 'np', 'pd'

Bundler.set_bundles_root('..', '..', 'bundles')
