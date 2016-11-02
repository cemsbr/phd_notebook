"""Functions used by different files."""
import numpy as np


def safe_log(df_in):
    """Log of zero is zero."""
    values = df_in.copy()
    values[values == 0] = 1
    return np.log(values)
