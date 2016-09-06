"""Detect outliers from parsed experiments."""
import pandas as pd

from lib.bundler import BaseBundle
from lib.outlier import Outlier


class Bundle(BaseBundle):
    """Add a boolean field `outlier` to the parsed data."""

    def __init__(self):
        """`basenames` for the parent constructor and dependencies."""
        super().__init__(['outliers_detected.csv', 'no_outliers.csv'])
        self.dependencies = ['wikipedia_profiling', 'wikipedia_target',
                             'hbsort_parsing', 'hbkmeans_parsing']

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
        dfs = []
        for bundle_name in self.dependencies:
            bundle = self.bundler.get_bundle(bundle_name)
            dfs.append(get_bundle_df(bundle))
        return pd.concat(dfs)


def add_outlier_column(df):
    """Add a boolean 'outlier' column in the end of DataFrame."""
    outlier = Outlier(df, col='duration_ms')
    last_col = len(df.columns) - 1
    df.insert(last_col, 'outlier', outlier.detect())


def get_bundle_df(bundle):
    """Return experiment results of a bundle."""
    dfs = [get_filename_df(f) for f in bundle.filenames]
    return dfs[0] if len(dfs) == 1 else pd.concat(dfs)


def get_filename_df(filename):
    """Return experiment result from a bundle's filename.

    Add two columns in the beginning:

        1. app: application name
        2. set: profiling or target
    """
    app, sset = filename.split('/')[-1][:-4].split('_')
    df = pd.read_csv(filename)
    df.insert(0, 'application', app)
    df.insert(1, 'set', sset)
    return df.rename(columns={'input_bytes': 'input',
                              'input_samples': 'input'})
