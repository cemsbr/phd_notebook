"""The one and only module for the sake of documentation."""
import numpy as np


class Outlier:
    """Detect outliers using interquartile range."""

    def __init__(self, df, m=1.5, col='ms'):
        """Detect outliers based on interquartile range.

        :param float m: IQR multiplier. Smaller values give more outliers.
        """
        self._df, self._m, self._col = df, m, col
        group_cols = df.columns.tolist()
        group_cols.remove(self._col)
        self._group_cols = group_cols
        self._outliers = self.detect()

    def remove(self):
        return self._df[~self._outliers]

    def get_overview(self):
        """Number of outliers by worker amount and input size.

        :param df: DataFrame with bool outlier column (see is_outlier())
        """
        df = self._df.copy()
        outcol = 'outliers'
        df[outcol] = self._outliers.astype('int')

        group = df.groupby(self._group_cols)
        overview = group.agg({self._col: np.size, outcol: np.sum})
        # Display number of available executions (non outliers)
        overview['~outliers'] = overview[self._col] - overview[outcol]
        overview['mean (sec)'] = self._df[~self._outliers].groupby(
            self._group_cols).mean()[self._col]/1000

        # Improving readability
        overview = overview.rename(columns={self._col: 'samples'})
        return overview[['mean (sec)', '~outliers', outcol, 'samples']]

    def detect(self):
        """Return a boolean column with True for outliers."""
        df, m = self._df, self._m
        group_values = df.groupby(self._group_cols)[self._col]
        lower = group_values.transform(
            lambda x: x.quantile(q=.25) - m * (x.quantile(q=.75) - x.quantile(q=.25)))
        upper = group_values.transform(
            lambda x: x.quantile(q=.75) + m * (x.quantile(q=.75) - x.quantile(q=.25)))
        return (df[self._col] < lower) | (df[self._col] > upper)
