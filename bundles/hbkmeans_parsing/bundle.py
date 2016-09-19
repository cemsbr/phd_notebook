"""Bundle to generate data from Spark log files, HiBench K-means app."""
from lib.bundler import BaseBundle
from lib.hbkmeans_parser import HBKmeansParser
from lib.csv_gen import CSVGen
from lib.parser import Parser
from sparklogstats import LogParser


class Bundle(BaseBundle):
    """Generate both profiling and target data from log files."""

    def __init__(self):
        """`basenames` for the parent constructor."""
        super().__init__('hbkmeans.csv')

    def run(self):
        """Parse logs and save relevant information in CSV files."""
        self.start()

        rows = HBKmeansParser.map(get_row)
        rows.sort()

        csv_gen = CSVGen()
        header = ('workers', 'set', 'input_samples', 'duration_ms',
                  'in_memory')
        writer = csv_gen.get_writer(header, self.filename)
        writer.writerows(rows)
        csv_gen.close()
        self.finish()


def get_row(log):
    """Return a row using only one LogParser instance."""
    parser = LogParser()
    app = parser.parse_file(log)
    samples = app.records_read
    sset = HBKmeansParser.get_set(samples)
    return (len(app.slaves), sset, samples, app.duration,
            Parser.fits_in_memory(app))


if __name__ == '__main__':
    Bundle().update()
