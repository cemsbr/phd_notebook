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
        header = ['workers', 'set', 'input_bytes', 'duration_ms',
                  'in_memory'] + get_stages()
        writer = csv_gen.get_writer(header, self.filename)

        for app in HBSortParser.get_apps():
            size = app.bytes_read
            sset = HBSortParser.get_set(size)
            # Use sum to count "sucessful_tasks" generator length
            tasks = [sum(1 for _ in s.successful_tasks) for s in app.stages]
            row = [len(app.slaves), sset, size, app.duration,
                   Parser.fits_in_memory(app)] + tasks
            writer.writerow(row)

        csv_gen.close()
        self.finish()


def get_stages():
    """Return stage names."""
    app = HBSortParser.get_app()
    total_stages = len(app.stages)
    return ['s{:02d}'.format(i) for i in range(total_stages)]


if __name__ == '__main__':
    Bundle().update()
