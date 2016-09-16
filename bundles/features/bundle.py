"""Calculate features."""
import pandas as pd

from lib.bundler import BaseBundle
from lib.config import Config
from lib.featurecreator import FeatureCreator


class Bundle(BaseBundle):
    """Calculate features."""

    def __init__(self):
        """Basename to be created."""
        super().__init__('features.csv')
        self.dependencies = 'outlier_detection',

    def run(self):
        """Evaluate models."""
        self.start()
        df = self._get_initial_df()
        self._get_features(df).to_csv(self.filename, index=False)
        self.finish()

    def _get_initial_df(self):
        """."""
        csv_file = self.bundler.get_bundle('outlier_detection').get_file(
            'no_outliers.csv')
        return pd.read_csv(csv_file)

    @classmethod
    def _get_features(cls, df):
        creator = FeatureCreator(df)
        creator.add_cols(Config.EXTRA_FEATURES)
        return creator.expand_poly(Config.DEGREE, Config.NO_EXPANSION,
                                   Config.DEL_FEATURES)


if __name__ == '__main__':
    Bundle().update()
