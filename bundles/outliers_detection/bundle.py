"""Detect outliers from parsed experiments."""
from lib.bundler import BaseBundle


class Bundler(BaseBundle):
    """Add a boolean field `outlier` to the parsed data."""

    def __init__(self):
        """Define `version` and `filenames` for the parent constructor."""
        version = 1
        filenames = ('outliers_detected.csv')
        super().__init__(version, filenames)
        self.add_dependency('wikipedia_profiling', 'wikipedia_target',
                            'hbsort_parsing', 'hbkmeans_parsing')

    def run(self):
        """Generate the output file."""
        pass
