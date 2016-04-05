"""The one and only module for the sake of documentation."""
import numpy as np


class Outlier:
    """Detect outliers using interquartile range."""

    @staticmethod
    def get_overview(df):
        """Number of outliers by worker amount and input size.

        :param df: DataFrame with bool outlier column (see is_outlier())
        """
        dfc = df.copy()
        dfc['outlier'] = dfc['outlier'].astype('int')

        group = dfc.groupby(['input', 'workers'])
        overview = group.agg({'ms': np.size, 'outlier': np.sum})
        # Display number of available executions to be analyzed
        overview['available'] = overview.ms - overview.outlier

        return overview.rename(columns={'ms': 'samples',
                                        'outlier': 'outliers'})

    @staticmethod
    def is_outlier(df, m=1.5):
        """Return a boolean column with True for outliers.

        :param float m: IQR multiplier. Smaller values give more outliers.
        """
        group = df.groupby(['input', 'workers'])
        lower = group.ms.transform(
            lambda x: x.quantile(q=.25) - m * (x.quantile(q=.75) - x.quantile(q=.25)))
        upper = group.ms.transform(
            lambda x: x.quantile(q=.75) + m * (x.quantile(q=.75) - x.quantile(q=.25)))
        return (df.ms < lower) | (df.ms > upper)
