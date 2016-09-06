"""Parse all log files of HiBench Sort application."""
import csv
from os.path import dirname, join

from lib.bundler import BaseBundle
from sparklogstats import LogParser


class Bundle(BaseBundle):
    """Parse both profiling and target logs."""

    def __init__(self):
        """Define `basenames` for the parent constructor."""
        super().__init__(['hbsort_profiling.csv', 'hbsort_target.csv'])

    def run(self):
        """Parse logs and extract relevant information."""
        self.start()

        root = join(dirname(__file__), '..', '..')
        folder = join(root, 'data', 'hibench', 'sort', '*')
        parser = LogParser()
        apps = parser.parse_folder(folder)

        # CSV files
        files = [open(f, 'w', newline='') for f in self.filenames]
        prof_csv, tgt_csv = [csv.writer(f) for f in files]

        # Header
        for writer in [prof_csv, tgt_csv]:
            writer.writerow(['workers', 'input_bytes', 'duration_ms'])

        for app in apps:
            size = app.bytes_read
            if len(app.slaves) == 123 or len(app.slaves) == 12:
                continue
            row = (len(app.slaves), size, app.duration)
            if size >= 3284983900:
                tgt_csv.writerow(row)
            else:
                prof_csv.writerow(row)

        for file in files:
            file.close()

        self.finish()

if __name__ == '__main__':
    Bundle().update()
