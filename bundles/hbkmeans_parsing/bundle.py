"""Bundle to generate data from Spark log files, HiBench K-means app."""
import csv
from os.path import dirname, join

from lib.bundler import BaseBundle
from sparklogstats import LogParser


class Bundle(BaseBundle):
    """Generate both profiling and target data from log files."""

    def __init__(self):
        """`basenames` for the parent constructor."""
        super().__init__(['hbkmeans_profiling.csv', 'hbkmeans_target.csv'])

    def run(self):
        """Parse logs and save relevant information in CSV files."""
        self.start()

        root = join(dirname(__file__), '..', '..')
        folder = join(root, 'data', 'hibench', 'kmeans')
        apps = LogParser().parse_folder(folder)
        out_files = [open(f, 'w', newline='')
                     for f in self.filenames]
        csvs = [csv.writer(f) for f in out_files]
        prof_csv, tgt_csv = csvs

        header = ['workers', 'input_samples', 'duration_ms']
        for f in csvs:
            f.writerow(header)

        for app in apps:
            records = app.records_read
            row = (len(app.slaves), records, app.duration)
            if records >= 16384000:
                tgt_csv.writerow(row)
            else:
                prof_csv.writerow(row)

        for out_file in out_files:
            out_file.close()

        self.finish()


if __name__ == '__main__':
    Bundle().update()
