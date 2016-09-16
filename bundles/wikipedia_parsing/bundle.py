"""Bundle to generate data from Spark log files."""
from itertools import cycle

from lib.bundler import BaseBundle
from bundles.common_parsing import CSVGen, Parser


class Bundle(BaseBundle):
    """Bundle for wikipedia target data.

    Input sizes are not correct in the log files, so they are hard coded.
    """

    def __init__(self):
        """Define `basenames` for the parent constructor."""
        super().__init__('wikipedia.csv')
        self._parser = Parser()
        self._writer = None

    def run(self):
        """Parse logs and extract relevant information."""
        self.start()
        header = ('workers', 'set', 'input_bytes', 'duration_ms', 'in_memory')
        csvgen = CSVGen()
        self._writer = csvgen.get_writer(header, self.filename)
        self._parse_profiling_set()
        self._parse_target_set()
        csvgen.close()
        self.finish()

    def _parse_profiling_set(self):
        """Parse profiling set."""
        # single-VM experiment
        apps = self._parser.get_apps('wikipedia', 'profiling', 'one_vm')
        self._parse(apps, 'profiling', _get_input_sizes_1vm())

        # Strong-scaling experiment
        all_apps = self._parser.get_apps('wikipedia', 'profiling',
                                         'strong_scaling1')
        # Use 1 slave from 'one_vm' and skip these ones.
        apps = (a for a in all_apps if len(a.slaves) > 1)
        self._parse(apps, 'profiling', cycle((1073741798,)))

    def _parse_target_set(self):
        apps = self._parser.get_apps('wikipedia', 'target')
        self._parse(apps, 'target', cycle((48542876756,)))

    def _parse(self, apps, sset, sizes):
        for app, size in zip(apps, sizes):
            self._writer.writerow((len(app.slaves), sset, size, app.duration,
                                   self._parser.fits_in_memory(app)))


def _get_input_sizes_1vm():
    """Return a generator for input sizes ordered by filename."""
    for size in (134217465, 268425644, 536816741, 1073741798):
        for _ in range(10):
            yield size


if __name__ == '__main__':
    Bundle().update()
