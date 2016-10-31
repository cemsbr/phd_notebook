"""Parse all log files of HiBench Sort application."""
from lib.bundler import BaseBundle
from lib.csv_gen import CSVGen
from lib.hbsort_parser import HBSortParser
from lib.parser import Parser
from lib.stage_stats import StageStats


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
        app = HBSortParser.get_app()
        stage_titles = StageStats.get_titles(app.stages)
        header = ['workers', 'set', 'input_bytes', 'input_records',
                  'duration_ms', 'in_memory'] + stage_titles
        writer = csv_gen.get_writer(header, self.filename)

        for app in HBSortParser.get_apps():
            size = app.bytes_read
            sset = HBSortParser.get_set(size)
            stage_stats = StageStats.get_stats(app.stages)
            row = [len(app.slaves), sset, size, app.records_read, app.duration,
                   Parser.fits_in_memory(app)] + stage_stats
            writer.writerow(row)

        csv_gen.close()
        self.finish()


if __name__ == '__main__':
    Bundle().update()
