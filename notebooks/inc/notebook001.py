"""Shorten Cache Effect notebook code."""
# Disable W0401 Wildcard import pd_settings
#         W0614 Unused import matplotlib from wildcard import
#         W0611 Unused Parser imported from lib.parser as SparkParser
# pylint: disable=W0401,W0611,W0614
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import inc.pd_config
import inc.plt_config  # noqa
from lib.parser import Parser as SparkParser

__all__ = ('display', 'np', 'pd', 'plt', 'SparkParser')
