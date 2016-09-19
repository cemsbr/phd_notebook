"""Parse all log files of HiBench Sort application."""
from lib.bundler import BaseBundle
from lib.csv_gen import CSVGen
from lib.hbsort_parser import HBSortParser
from lib.parser import Parser


class Bundle(BaseBundle):
    """Parse both profiling and target logs."""

    def __init__(self):
        """Define `basenames` for the parent constructor."""
        super().__init__('hbsort.csv')

    def run(self):
        """Parse logs and extract relevant information."""
        self.start()

        # CSV files
        csv_gen = CSVGen()
        header = ('workers', 'set', 'input_bytes', 'duration_ms', 'in_memory')
        writer = csv_gen.get_writer(header, self.filename)

        for app in HBSortParser.get_apps():
            size = app.bytes_read
            sset = HBSortParser.get_set(size)
            row = (len(app.slaves), sset, size, app.duration,
                   Parser.fits_in_memory(app))
            writer.writerow(row)

        csv_gen.close()
        self.finish()


if __name__ == '__main__':
    Bundle().update()
