"""Detect outliers from parsed experiments."""
import pandas as pd

from lib.bundler import BaseBundle
from lib.outlier import Outlier


class Bundle(BaseBundle):
    """Add a boolean field `outlier` to the parsed data."""

    def __init__(self):
        """`basenames` for the parent constructor and dependencies."""
        super().__init__('outliers_detected.csv', 'no_outliers.csv')
        self.dependencies = ['only_memory']

    def run(self):
        """Generate the output file."""
        self.start()
        df = self.get_dataframe()
        add_outlier_column(df)
        df.to_csv(self.filenames[0], index=False)
        df[~df.outlier].drop('outlier', axis=1).to_csv(self.filenames[1],
                                                       index=False)
        self.finish()

    def get_dataframe(self):
        """Get dataframe with all parsed experiment results."""
        filename = self.bundler.get_bundle('only_memory').filename
        return pd.read_csv(filename)


def add_outlier_column(df):
    """Add a boolean 'outlier' column in the end of DataFrame."""
    outlier = Outlier(df, col='duration_ms')
    last_col = len(df.columns) - 1
    df.insert(last_col, 'outlier', outlier.detect())


if __name__ == '__main__':
    Bundle().update()
