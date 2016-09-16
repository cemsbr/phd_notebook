"""Join all parsing results into one file."""
import pandas as pd
from lib.bundler import BaseBundle


class Bundle(BaseBundle):
    """Join all parsing results into one file."""

    def __init__(self):
        """Setup output filename and dependencies."""
        super().__init__('all_applications.csv')
        self.dependencies = ('wikipedia_parsing', 'hbsort_parsing',
                             'hbkmeans_parsing')

    def run(self):
        """Generate CSV output with additional columns.

        Add application as the first column
        """
        self.start()
        # Load both profiling and target sets at once.
        self.get_dataframe().to_csv(self.filename, index=False)
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
    df = pd.read_csv(filename)
    df.insert(0, 'application', app)
    return df.rename(columns={'input_bytes': 'input',
                              'input_samples': 'input'})


if __name__ == '__main__':
    Bundle().update()
