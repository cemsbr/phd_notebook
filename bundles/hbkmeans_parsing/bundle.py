"""Bundle to generate data from Spark log files, HiBench K-means app."""
import glob
from multiprocessing import Pool
from os import path
from random import shuffle

from lib.bundler import BaseBundle
from bundles.common_parsing import CSVGen, Parser
from sparklogstats import LogParser


class Bundle(BaseBundle):
    """Generate both profiling and target data from log files."""

    def __init__(self):
        """`basenames` for the parent constructor."""
        super().__init__('hbkmeans.csv')

    def run(self):
        """Parse logs and save relevant information in CSV files."""
        self.start()

        root = path.join(path.dirname(__file__), '..', '..')
        filenames = path.join(root, 'data', 'hibench', 'kmeans', 'app-*')
        logs = sorted(glob.glob(filenames))
        shuffle(logs)  # last logs are longer
        with Pool() as p:
            rows = p.map(get_row, logs)
        rows.sort()

        csvgen = CSVGen()
        header = ('workers', 'set', 'input_samples', 'duration_ms',
                  'in_memory')
        writer = csvgen.get_writer(header, self.filename)
        writer.writerows(rows)
        csvgen.close()
        self.finish()


def get_row(log):
    """Return a row using only one LogParser instance."""
    parser = LogParser()
    app = parser.parse_file(log)
    records = app.records_read
    sset = 'profiling' if records < 16384000 else 'target'
    return (len(app.slaves), sset, records, app.duration,
            Parser.fits_in_memory(app))


if __name__ == '__main__':
    Bundle().update()
