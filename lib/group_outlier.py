"""Outlier detection by groups."""
import pandas as pd


class GroupOutlier:
    """Outlier detection by groups."""

    _groups = None

    @classmethod
    def set_groups(cls, groups):
        """Set groups for outlier detection.

        Args:
            groups (pd.DataFrame): Groups for outlier detection.
        """
        cls._groups = groups

    @classmethod
    def is_outlier(cls, df, col, m=1.5):
        """Return a binary array answering if row is an outlier."""
        grouped = cls._get_grouped_df(df, col)
        lower, upper = GroupOutlier._get_range(grouped, m)
        return (df[col] < lower) | (df[col] > upper)

    @classmethod
    def remove_outliers(cls, df, col, m=1.5):
        """Remove outliers from *df* using groups from :meth:`set_groups`."""
        is_outlier = cls.is_outlier(df, col, m)
        return df[~is_outlier]

    @classmethod
    def _get_grouped_df(cls, df, col):
        df_group = pd.concat([df[col], cls._groups], axis=1, join='inner')
        assert len(df_group) == len(df)
        return df_group.groupby('group')[col]

    @staticmethod
    def _get_range(grouped, m):
        def iqr(values):
            """Return the interquartile range."""
            return values.quantile(q=.75) - values.quantile(q=.25)

        lower = grouped.transform(lambda x: x.quantile(q=.25) - m * iqr(x))
        upper = grouped.transform(lambda x: x.quantile(q=.75) + m * iqr(x))
        return lower, upper
