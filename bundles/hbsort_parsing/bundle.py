"""Parse all log files of HiBench Sort application."""
from lib.bundler import BaseBundle
from bundles.common_parsing import CSVGen, Parser


class Bundle(BaseBundle):
    """Parse both profiling and target logs."""

    def __init__(self):
        """Define `basenames` for the parent constructor."""
        super().__init__('hbsort.csv')

    def run(self):
        """Parse logs and extract relevant information."""
        self.start()

        parser = Parser()
        apps = parser.get_apps('hibench', 'sort', '*')

        # CSV files
        csvgen = CSVGen()
        header = ('workers', 'set', 'input_bytes', 'duration_ms', 'in_memory')
        writer = csvgen.get_writer(header, self.filename)

        for app in apps:
            if len(app.slaves) == 123 or len(app.slaves) == 12:
                continue
            size = app.bytes_read
            sset = 'profiling' if size < 3284983900 else 'target'
            row = (len(app.slaves), sset, size, app.duration,
                   parser.fits_in_memory(app))
            writer.writerow(row)

        csvgen.close()
        self.finish()


if __name__ == '__main__':
    Bundle().update()
