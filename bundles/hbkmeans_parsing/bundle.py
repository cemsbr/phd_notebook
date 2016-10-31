"""Bundle to generate data from Spark log files, HiBench K-means app."""
from lib.bundler import BaseBundle
from lib.csv_gen import CSVGen
from lib.hbkmeans_parser import HBKmeansParser
from lib.parser import Parser
from lib.stage_stats import StageStats
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
        app = HBKmeansParser.get_app()
        stage_titles = StageStats.get_titles(app.stages)
        header = ['workers', 'set', 'input_bytes', 'input_records',
                  'duration_ms', 'in_memory'] + stage_titles
        writer = csv_gen.get_writer(header, self.filename)
        writer.writerows(rows)
        csv_gen.close()
        self.finish()


def get_row(log):
    """Return a row using only one LogParser instance."""
    parser = LogParser()
    app = parser.parse_file(log)
    input_bytes = app.bytes_read
    input_records = app.records_read
    sset = HBKmeansParser.get_set(input_records)

    stage_stats = StageStats.get_stats(app.stages)
    return [len(app.slaves), sset, input_bytes, input_records, app.duration,
            Parser.fits_in_memory(app)] + stage_stats


if __name__ == '__main__':
    Bundle().update()
