"""Bundle to generate data from Spark log files."""
import csv
from os.path import dirname, join

from lib.bundler import BaseBundle
from sparklogstats import LogParser


class Bundle(BaseBundle):
    """Bundle for wikipedia target data.

    Input sizes are not correct in the log files, so they are hard coded.
    """

    def __init__(self):
        """Define `basenames` for the parent constructor."""
        super().__init__(['wikipedia_profiling.csv'])
        self._parser = None
        self._wiki_folder = None

    def run(self):
        """Parse logs and extract relevant information."""
        self.start()

        self._parser = LogParser()
        self._wiki_folder = _get_wiki_folder()

        output = self.filenames[0]
        with open(output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['workers', 'input_bytes', 'duration_ms'])
            # single-VM experiment
            self._parse('one_vm', _get_input_sizes_1vm(), writer)
            # Strong-scaling experiment
            self._parse('strong_scaling1', (1073741798 for _ in range(30)),
                        writer)

        self.finish()

    def _parse(self, folder, sizes, writer):
        input_folder = join(self._wiki_folder, folder)
        apps = self._parser.parse_folder(input_folder)
        for app, size in zip(apps, sizes):
            # Use only 1-VM logs from experiment "one_vm".
            if folder == 'one_vm' or len(app.slaves) > 1:
                writer.writerow((len(app.slaves), size, app.duration))


def _get_input_sizes_1vm():
    """Return a generator for input sizes ordered by filename."""
    for size in (134217465, 268425644, 536816741, 1073741798):
        for _ in range(10):
            yield size


def _get_wiki_folder():
    """Return folder of wikipedia experiment logs."""
    root = join(dirname(__file__), '..', '..')
    return join(root, 'data', 'wikipedia', 'profiling')


if __name__ == '__main__':
    Bundle().update()
