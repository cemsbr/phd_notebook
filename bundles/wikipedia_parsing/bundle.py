"""Bundle to generate data from Spark log files."""
from lib.bundler import BaseBundle
from lib.csv_gen import CSVGen
from lib.parser import Parser
from lib.stage_stats import StageStats
from lib.wikipedia_parser import WikipediaParser


class Bundle(BaseBundle):
    """Bundle for wikipedia target data.

    Input sizes are not correct in the log files, so they are hard coded.
    """

    def __init__(self):
        """Define `basenames` for the parent constructor."""
        super().__init__('wikipedia.csv')
        self._parser = WikipediaParser()
        self._writer = None

    def run(self):
        """Parse logs and extract relevant information."""
        self.start()
        app = self._parser.get_app()
        stage_titles = StageStats.get_titles(app.stages)
        header = ['workers', 'set', 'input_bytes', 'input_records',
                  'duration_ms', 'in_memory'] + stage_titles
        csv_gen = CSVGen()
        self._writer = csv_gen.get_writer(header, self.filename)
        self._write_sets()
        csv_gen.close()
        self.finish()

    def _write_sets(self):
        """Parse profiling set."""
        # Profiling set
        apps_sizes = self._parser.get_profiling()
        self._write_set('profiling', apps_sizes)

        # Target set
        apps_sizes = self._parser.get_target()
        self._write_set('target', apps_sizes)

    def _write_set(self, sset, apps_sizes):
        for app, size in apps_sizes:
            stage_stats = StageStats.get_stats(app.stages)
            self._writer.writerow([len(app.slaves), sset, size,
                                   app.records_read, app.duration,
                                   Parser.fits_in_memory(app)] + stage_stats)


if __name__ == '__main__':
    Bundle().update()
