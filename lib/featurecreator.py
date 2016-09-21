"""Create extra columns with features."""
from itertools import combinations_with_replacement as comb_repl, chain
import numpy as np
import pandas as pd


class FeatureCreator:
    """Create extra columns with features."""

    def __init__(self, df):
        """Source DataFrame to be modified.

        If you don't want to modify it, provide ``df.copy()``.

        Args:
            df (pd.DataFrame): Source DataFrame.
        """
        self._df = df

    def add_cols(self, cols):
        """Add features to DataFrame.

        Args:
            cols (list): 2-tuples with column name and function that receive
                constructor's DataFrame and returns new values.

        Returns:
            pd.DataFrame: constructor's DataFrame with added columns.
        """
        for name, fn in cols:
            if name not in self._df.columns:
                self._df[name] = fn(self._df)
        return self._df

    def expand_poly(self, degree, exclude_before, exclude_after):
        """Expand features to `degree`.

        Args:
            degree (int): Polynomial degree.
            exclude_before (list): Column names to exclude before expansion.
            exclude_after (list): Column names to remove from final result,
                after expansion.

        Returns:
            pd.DataFrame: constructor's DataFrame with added columns.
        """
        exclude_df = self._df[exclude_before]
        self._df = self._df.drop(exclude_before, axis=1)

        nr_cols = self._df.shape[1]
        for powers in _get_all_powers(nr_cols, degree):
            col = self._get_new_col_name(powers)
            if col not in exclude_after:
                value = self._get_new_col_value(powers)
                self._df[col] = value

        self._df = pd.concat([exclude_df, self._df], axis=1)
        return self._df

    def _get_new_col_name(self, powers):
        cols = []
        for df_col, power in zip(self._df.columns, powers):
            if power > 0:
                col = df_col
                if power > 1:
                    col = '({})^{:d}'.format(col, power)
                cols.append(col)
        return ' * '.join(cols)

    def _get_new_col_value(self, powers):
        value = np.ones(len(self._df))
        for col, power in zip(self._df.columns, powers):
            if power > 0:
                value *= self._df[col]**power
        return value


def _get_all_powers(n, degree):
    r"""Return the powers of each column which sum `degree`.

    Args:
        n (int): Number of columns.
        degree (int): Maximum sum of column degrees.

    Example:
        >>> import pandas as pd
        >>> from lib.featurecreator import FeatureCreator
        >>> df = pd.DataFrame([[1, 2]])  # two-column DataFrame
        >>> powers1 = FeatureCreator(df)._get_all_powers(1)
        >>> list(powers1)
        [array([1, 0]), array([0, 1])][array([1, 0]), array([0, 1])]
        >>> powers2 = FeatureCreator(df)._get_all_powers(2)
        >>> list(powers2)
        [array([1, 0]), array([0, 1]), array([2, 0]), array([1, 1]), \
        array([0, 2])]

    Return:
        generator: arrays with powers ordered by column.
    """
    powers = chain.from_iterable(comb_repl(
        range(n), d) for d in range(1, degree + 1))
    return (np.bincount(p, minlength=n) for p in powers)
