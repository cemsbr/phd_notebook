"""Join all parsing results into one file."""
import pandas as pd
from lib.bundler import BaseBundle


class Bundle(BaseBundle):
    """Join all parsing results into one file.

    Application name is taken from the filename.
    """

    def __init__(self):
        """Setup output filename and dependencies."""
        super().__init__('all_applications.csv')
        self.dependencies = ('wikipedia_parsing', 'hbsort_parsing',
                             'hbkmeans_parsing')

    def run(self):
        """Generate CSV output with additional columns.

        Add application as the first column.
        """
        self.start()
        # Load both profiling and target sets at once.
        apps_df = self.get_dataframe()
        apps_df = sort_columns(apps_df)
        apps_df.to_csv(self.filename, index=False)
        self.finish()

    def get_dataframe(self):
        """Get dataframe with all parsed experiment results."""
        dfs = []
        for bundle_name in self.dependencies:
            bundle = self.bundler.get_bundle(bundle_name)
            dfs.append(get_application_df(bundle.filename))
        return pd.concat(dfs)


def get_application_df(filename):
    """Return experiment result from a bundle's filename.

    Assume that `filename` has the form `(*/)application.csv`.
    `input_bytes` and `input_records` become `input` and application column.
    """
    app = filename.split('/')[-1][:-4]
    # If not object, integers will become floats because of NaN values
    df = pd.read_csv(filename, dtype=object)
    df.insert(0, 'application', app)
    return df.rename(columns={'input_bytes': 'input',
                              'input_records': 'records'})


def sort_columns(df):
    """Reorder columns."""
    cols = df.columns.tolist()
    first_cols = ('application', 'set', 'in_memory', 'workers', 'input',
                  'records', 'duration_ms')
    for col in first_cols[::-1]:
        cols.remove(col)
        cols.insert(0, col)
    return df.reindex_axis(cols, axis=1, copy=False)


if __name__ == '__main__':
    Bundle().update()
