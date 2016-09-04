"""Bundle to generate data from Spark log files."""
import csv
from os.path import dirname, join

from lib.bundler import BaseBundle
from sparklogstats import LogParser


class Bundle(BaseBundle):
    """Bundle for wikipedia target data."""

    def __init__(self):
        """`version` and `filenames` for the parent constructor."""
        version = 1
        filenames = ('wikipedia_target.csv',)
        super().__init__(version, filenames)

    def run(self):
        """Parse logs and extract relevant information."""
        self.start()

        # Input
        root = join(dirname(__file__), '..', '..')
        input_folder = join(root, 'data', 'wikipedia', 'target')
        apps = LogParser().parse_folder(input_folder)

        # Output
        output = self.get_versioned_filename(self.filenames[0])
        with open(output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['workers', 'input_bytes', 'duration_ms'])
            for app in apps:
                # Spark log incorrectly reports bytes read, though records read
                # is correct.
                writer.writerow((len(app.slaves), 48542876756, app.duration))

        self.finish()


if __name__ == '__main__':
    Bundle().update()
