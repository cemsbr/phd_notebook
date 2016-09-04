"""Parse all log files of HiBench Sort application."""
import csv
from os.path import dirname, join

from lib.bundler import BaseBundle
from sparklogstats import LogParser


class Bundle(BaseBundle):
    """Parse both profiling and target logs."""

    def __init__(self):
        """Define `version` and `filenames` for the parent constructor."""
        version = 1
        filenames = ('hbsort_profiling.csv', 'hbsort_target.csv')
        super().__init__(version, filenames)

    def run(self):
        """Parse logs and extract relevant information."""
        self.start()

        root = join(dirname(__file__), '..', '..')
        folder = join(root, 'data', 'hibench', 'sort')
        filenames = self.get_versioned_filenames()
        parser = LogParser()
        for subfolder, filename in zip(['profiling', 'target*'], filenames):
            folder2parse = join(folder, subfolder)
            apps = parser.parse_folder(folder2parse)
            _parse_folder(apps, filename)

        self.finish()


def _parse_folder(apps, out_filename):
    """Generate the target or profiling output."""
    with open(out_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['workers', 'input_bytes', 'duration_ms'])
        for app in apps:
            row = (len(app.slaves), app.bytes_read, app.duration)
            writer.writerow(row)


if __name__ == '__main__':
    Bundle().update()
